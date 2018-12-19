/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#else
#include <sys/socket.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <event2/event.h>
#include <event2/http.h>
#include <event2/buffer.h>
#include <event2/util.h>
#include <event2/keyvalq_struct.h>


#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ext_list.hpp>

#include "serialization.h"
#include "cnn.hpp"
#include "detector.hpp"
//#include "face_reid.hpp"
void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec);

using namespace InferenceEngine;

struct Context {
    detection::FaceDetection *face_detector;
    VectorCNN *face_reid;
    VectorCNN *landmarks_detector;
};

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
static std::string guess_plugin_path(void) {
    const char* p;
    if ((p = getenv("IE_PLUGINS_PATH")))
        return p;

    Dl_info info;
    if (dladdr((void*)GetInferenceEngineVersion, &info) &&
        (p = strrchr(info.dli_fname, '/'))) {
        return std::string(info.dli_fname, p - info.dli_fname);
    }
    return "";
}

#ifndef UNUSED  // sample/common.hpp is not included
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}
#endif

/* Callback used for the /dump URI, and for every non-GET request:
 * dumps all information to stdout and gives back a trivial 200 ok */
static void
dump_request_cb(struct evhttp_request *req, void *arg)
{
    const char *cmdtype;
    struct evkeyvalq *headers;
    struct evkeyval *header;
    struct evbuffer *buf;

    switch (evhttp_request_get_command(req)) {
        case EVHTTP_REQ_GET: cmdtype = "GET"; break;
        case EVHTTP_REQ_POST: cmdtype = "POST"; break;
        case EVHTTP_REQ_HEAD: cmdtype = "HEAD"; break;
        case EVHTTP_REQ_PUT: cmdtype = "PUT"; break;
        case EVHTTP_REQ_DELETE: cmdtype = "DELETE"; break;
        case EVHTTP_REQ_OPTIONS: cmdtype = "OPTIONS"; break;
        case EVHTTP_REQ_TRACE: cmdtype = "TRACE"; break;
        case EVHTTP_REQ_CONNECT: cmdtype = "CONNECT"; break;
        case EVHTTP_REQ_PATCH: cmdtype = "PATCH"; break;
        default: cmdtype = "unknown"; break;
    }

    printf("Received a %s request for %s\nHeaders:\n",
        cmdtype, evhttp_request_get_uri(req));

    headers = evhttp_request_get_input_headers(req);
    for (header = headers->tqh_first; header;
        header = header->next.tqe_next) {
        printf("  %s: %s\n", header->key, header->value);
    }

    buf = evhttp_request_get_input_buffer(req);
    puts("Input data: <<<");
    while (evbuffer_get_length(buf)) {
        int n;
        char cbuf[128];
        n = evbuffer_remove(buf, cbuf, sizeof(cbuf));
        if (n > 0)
            (void) fwrite(cbuf, 1, n, stdout);
    }
    puts(">>>");

    evhttp_send_reply(req, 200, "OK", NULL);
}

static void
info_request_cb(struct evhttp_request *req, void *arg)
{
    struct Context* context = (struct Context*)arg;
    if (evhttp_request_get_command(req) != EVHTTP_REQ_GET) {
        evhttp_add_header(evhttp_request_get_output_headers(req),
                          "Allow", "HEAD, GET");
        evhttp_send_error(req, 405, "Method Not Allowed");
        return;
    }

    std::ostringstream ss;
    ss << context->face_detector->net_.GetInputsInfo() << '\n'
       << context->face_reid->executable_network_.GetOutputsInfo() << '\n';

    struct evbuffer *out = evbuffer_new();
    evbuffer_add_printf(out, "%s", ss.str().c_str());
    evhttp_add_header(evhttp_request_get_output_headers(req),
        "Content-Type", "text/plain");
    evhttp_send_reply(req, 200, "OK", out);
    evbuffer_free(out);
}

static void
infer_request_cb(struct evhttp_request *req, void *arg)
{
    struct Context* context = (struct Context*)arg;
    if (evhttp_request_get_command(req) != EVHTTP_REQ_POST) {
        dump_request_cb(req, arg);
        return;
    }

    // read request body and decode
    struct evbuffer *buf = evhttp_request_get_input_buffer(req);
    cv::Mat image(1, evbuffer_get_length(buf), CV_8UC1);
    int n = evbuffer_remove(buf, image.data, image.total()); 
    assert(n == image.total());
    image = cv::imdecode(image, cv::IMREAD_UNCHANGED);
    assert(image.data);

    // face detection
    detection::DetectedObjects faces;
    if (context->face_detector) {
        context->face_detector->enqueue(image);
        context->face_detector->submitRequest();
        context->face_detector->wait();
        context->face_detector->fetchResults();
        faces.swap(context->face_detector->results);
    } else {
        faces.emplace_back(cv::Rect(cv::Point(), image.size()), 1.0);
    }

    std::vector<cv::Mat> face_rois, landmarks, embeddings;
    for (const auto& face : faces) {
        //face_rois.push_back(image(face.rect));
        face_rois.emplace_back(image, face.rect);
    }
    context->landmarks_detector->Compute(face_rois, &landmarks, cv::Size(2, 5));
    AlignFaces(&face_rois, &landmarks);
    context->face_reid->Compute(face_rois, &embeddings);

    // generate and send response
    struct evbuffer *out = evbuffer_new();
    for (int k = 0; k < embeddings.size(); ++k) {
        const cv::Mat& m = embeddings[k];
        evbuffer_add_printf(out, "[%d]:", k);
        for (int i = 0; i < m.rows; ++i) {
            for(int j = 0; j < m.cols; ++j)
                evbuffer_add_printf(out, " %f", m.at<float>(i, j));
        }
        evbuffer_add_printf(out, "\n");
    }
    evbuffer_add_printf(out, "\n");
    evhttp_add_header(evhttp_request_get_output_headers(req),
        "Content-Type", "text/plain");
    evhttp_send_reply(req, 200, "OK", out);
    evbuffer_free(out);
}

static void
test_request_cb(struct evhttp_request *req, void *arg)
{
    struct evbuffer *evb = evbuffer_new();
    evbuffer_add_printf(evb, "It works!\n");
    evhttp_add_header(evhttp_request_get_output_headers(req),
        "Content-Type", "text/html");
    evhttp_send_reply(req, 200, "OK", evb);
    evbuffer_free(evb);

    //evhttp_send_error(req, 404, "File Not Found");
}

int
main(int argc, char **argv)
{
    in_port_t FLAGS_p = 1080;
    std::string FLAGS_c;
    std::string FLAGS_l;
    double FLAGS_exp_r_fd = 1.15;
    double FLAGS_t_fd = 0.6;
    int32_t FLAGS_inh_fd = 600;
    int32_t FLAGS_inw_fd = 600;
    std::string FLAGS_d_fd = "CPU";
    std::string FLAGS_d_lm = "CPU";
    std::string FLAGS_d_reid = "CPU";
    std::string::size_type p;

    int opt;
    while ((opt = getopt(argc, argv, "c:d:e:l:p:s:t:h")) != -1) {
        switch (opt) {
           case 'c': // gpu custom kernel
               FLAGS_c = optarg;
               break;
           case 'd': // device
               FLAGS_d_fd = optarg;
               if ((p = FLAGS_d_fd.find(',')) != std::string::npos) {
                   FLAGS_d_lm = FLAGS_d_fd.substr(p + 1);
                   FLAGS_d_fd.resize(p);
                   if ((p = FLAGS_d_lm.find(',')) != std::string::npos) {
                       FLAGS_d_reid = FLAGS_d_lm.substr(p + 1);
                       FLAGS_d_lm.resize(p);
                   }
               }
               break;
           case 'e': // expand
               FLAGS_exp_r_fd = atof(optarg);
               break;
           case 'l': // cpu custom kernel library
               FLAGS_l = optarg;
               break;
           case 'p': // port to listen
               FLAGS_p = (ev_uint16_t)atoi(optarg);
               break;
           case 's': // input size: WxH
               FLAGS_inw_fd = strtoul(optarg, &optarg, 10);
               if (tolower(*optarg) == 'x')
                   FLAGS_inh_fd = strtoul(optarg + 1, NULL, 10);
               else
                   FLAGS_inh_fd = FLAGS_inw_fd;
               break;
           case 't': // threshold of face detection
               FLAGS_t_fd = atoi(optarg);
               break;
           case 'h': // help
           default:
               fprintf(stderr, "Usage: %s [options] [[fd_model] lm_model reid_model]\n"
                       "\t-c gpu_kernel\n"
                       "\t-d dev_fd[,dev_lm[,dev_reid]] (default: %s,%s,%s)\n"
                       "\t-e expand_ratio (default: %.2f\n"
                       "\t-l cpu_library\n"
                       "\t-p port (default: %d)\n"
                       "\t-s widthXheight (default: %dX%d)\n"
                       "\t-t threshold (default: %.2f)\n", argv[0],
                       FLAGS_d_fd.c_str(), FLAGS_d_lm.c_str(), FLAGS_d_reid.c_str(),
                       FLAGS_exp_r_fd, FLAGS_p, FLAGS_inw_fd, FLAGS_inh_fd, FLAGS_t_fd);
               exit(EXIT_FAILURE);
        }
    }

    std::string FLAGS_m_fd, FLAGS_m_lm, FLAGS_m_reid;
    if (argc >= optind + 2) {
        FLAGS_m_fd = argv[optind];
        FLAGS_m_lm = argv[optind + 1];
        if (argc == optind + 2) {
            FLAGS_m_reid.swap(FLAGS_m_lm);
            FLAGS_m_lm.swap(FLAGS_m_fd);
        } else {
            FLAGS_m_reid = argv[optind + 2];
        }
    } else {
        const char* cvsdk_dir = getenv("INTEL_CVSDK_DIR");
        if (!cvsdk_dir)
            cvsdk_dir = "/opt/intel/computer_vision_sdk";
        std::string models_dir = std::string(cvsdk_dir) + "/deployment_tools/intel_models";
        // TODO: prefer FP16 on GPU
        FLAGS_m_fd = models_dir + "/face-detection-retail-0004/FP32/face-detection-retail-0004.xml";
        FLAGS_m_lm = models_dir + "/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml";
        FLAGS_m_reid = models_dir + "/face-reidentification-retail-0071/FP32/face-reidentification-retail-0071.xml";
    }

    auto fd_model_path = FLAGS_m_fd;
    auto fd_weights_path = fileNameNoExt(fd_model_path) + ".bin";
    auto fr_model_path = FLAGS_m_reid;
    auto fr_weights_path = fileNameNoExt(fr_model_path) + ".bin";
    auto lm_model_path = FLAGS_m_lm;
    auto lm_weights_path = fileNameNoExt(lm_model_path) + ".bin";

    // Load plugins
    std::map<std::string, InferencePlugin> plugins_for_devices;
    std::vector<std::string> devices = {FLAGS_d_fd, FLAGS_d_lm, FLAGS_d_reid};
    for (const auto &device : devices) {
        if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
            continue;
        }
        auto path = guess_plugin_path();
        InferencePlugin plugin = PluginDispatcher({path, ""}).getPluginByDevice(device);
        if ((device.find("CPU") != std::string::npos)) {
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
            if (!FLAGS_l.empty()) {
                // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                plugin.AddExtension(extension_ptr);
                std::cout << "CPU Extension loaded: " << FLAGS_l << std::endl;
            }
        } else if (!FLAGS_c.empty()) {
            // Load Extensions for other plugins not CPU
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }
        plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
        plugins_for_devices[device] = plugin;
    }

    struct Context context = { nullptr, nullptr, nullptr };

    // Load face detector
    detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
    face_config.plugin = plugins_for_devices[FLAGS_d_fd];
    face_config.is_async = true;
    face_config.enabled = !fd_model_path.empty();
    face_config.confidence_threshold = FLAGS_t_fd;
    face_config.input_h = FLAGS_inh_fd;
    face_config.input_w = FLAGS_inw_fd;
    face_config.increase_scale_x = FLAGS_exp_r_fd;
    face_config.increase_scale_y = FLAGS_exp_r_fd;
    detection::FaceDetection face_detector(face_config);
    if (face_config.enabled)
        context.face_detector = &face_detector;

    // Load face reid
    CnnConfig reid_config(fr_model_path, fr_weights_path);
    reid_config.max_batch_size = 16;
    reid_config.enabled = face_config.enabled && !fr_model_path.empty() && !lm_model_path.empty();
    reid_config.plugin = plugins_for_devices[FLAGS_d_reid];
    VectorCNN face_reid(reid_config);
    context.face_reid = &face_reid;

    // Load landmarks detector
    CnnConfig landmarks_config(lm_model_path, lm_weights_path);
    landmarks_config.max_batch_size = 16;
    landmarks_config.enabled = face_config.enabled && reid_config.enabled && !lm_model_path.empty();
    landmarks_config.plugin = plugins_for_devices[FLAGS_d_lm];
    VectorCNN landmarks_detector(landmarks_config);
    context.landmarks_detector = &landmarks_detector;


	struct event_base *base;
	struct evhttp *http;
	struct evhttp_bound_socket *handle;

	ev_uint16_t port = FLAGS_p;
#ifdef _WIN32
	WSADATA WSAData;
	WSAStartup(0x101, &WSAData);
#else
	if (signal(SIGPIPE, SIG_IGN) == SIG_ERR)
		return (1);
#endif

	base = event_base_new();
	if (!base) {
		fprintf(stderr, "Couldn't create an event_base: exiting\n");
		return 1;
	}

	/* Create a new evhttp object to handle requests. */
	http = evhttp_new(base);
	if (!http) {
		fprintf(stderr, "couldn't create evhttp. Exiting.\n");
		return 1;
	}

	evhttp_set_cb(http, "/info", info_request_cb, &context);
	evhttp_set_cb(http, "/infer", infer_request_cb, &context);
	evhttp_set_cb(http, "/test", test_request_cb, NULL);
	evhttp_set_gencb(http, dump_request_cb, NULL);

	/* Now we tell the evhttp what port to listen on */
	handle = evhttp_bind_socket_with_handle(http, "0.0.0.0", port);
	if (!handle) {
		fprintf(stderr, "couldn't bind to port %d. Exiting.\n",
		    (int)port);
		return 1;
	}

	{
		/* Extract and display the address we're listening on. */
		struct sockaddr_storage ss;
		evutil_socket_t fd;
		ev_socklen_t socklen = sizeof(ss);
		char addrbuf[128];
		void *inaddr;
		const char *addr;
		int got_port = -1;
		fd = evhttp_bound_socket_get_fd(handle);
		memset(&ss, 0, sizeof(ss));
		if (getsockname(fd, (struct sockaddr *)&ss, &socklen)) {
			perror("getsockname() failed");
			return 1;
		}
		if (ss.ss_family == AF_INET) {
			got_port = ntohs(((struct sockaddr_in*)&ss)->sin_port);
			inaddr = &((struct sockaddr_in*)&ss)->sin_addr;
		} else if (ss.ss_family == AF_INET6) {
			got_port = ntohs(((struct sockaddr_in6*)&ss)->sin6_port);
			inaddr = &((struct sockaddr_in6*)&ss)->sin6_addr;
		} else {
			fprintf(stderr, "Weird address family %d\n",
			    ss.ss_family);
			return 1;
		}
		addr = evutil_inet_ntop(ss.ss_family, inaddr, addrbuf,
		    sizeof(addrbuf));
		if (addr) {
			printf("Listening on %s:%d\n", addr, got_port);
		} else {
			fprintf(stderr, "evutil_inet_ntop failed\n");
			return 1;
		}
	}

	event_base_dispatch(base);

	return 0;
}

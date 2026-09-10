// idf/demo/main/main.c — ESP32 smoke smoke demo: run one inference through the
// Ferrite device runtime on hardware (Section 4.5).
//
// The graph is built by hand (relu over a 2-element tensor) so the demo has
// no dependency on the host importer/compiler. It proves init -> plan -> walk
// works on the part. Replace the kernels with your compiled FEMD model header
// for the real workload.

#include <stdio.h>
#include <string.h>

#include "allocator.h"
#include "tensor.h"
#include "graph.h"
#include "memory_planner.h"
#include "engine.h"

#define W_BUF 4096
#define A_BUF 4096

void app_main(void) {
    static unsigned char weight_buf[W_BUF] __attribute__((aligned(64)));
    static unsigned char act_buf[A_BUF]    __attribute__((aligned(64)));

    FeGraph g;
    fe_graph_init(&g);

    int s_in[] = {1, 2};
    int s_out[] = {1, 2};
    int t_in   = fe_graph_add_tensor(&g, "input",  DTYPE_FLOAT32, 2, s_in,  0);
    int t_out  = fe_graph_add_tensor(&g, "output", DTYPE_FLOAT32, 2, s_out, 0);

    int relu_in[]  = {t_in};
    int relu_out[] = {t_out};
    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL,     0, &t_in, 1);
    fe_graph_add_node(&g, "relu",   FE_OP_RELU,   relu_in,  1, relu_out, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_out,   1, NULL, 0);

    FeRuntime rt;
    if (fe_runtime_init(&rt, &g, weight_buf, W_BUF, act_buf, A_BUF)
        != FE_OK) {
        printf("Ferrite: runtime init failed\n");
        return;
    }
    if (fe_runtime_alloc_weights(&rt) != FE_OK) {
        printf("Ferrite: weight alloc failed\n");
        return;
    }

    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float in_data[] = { -1.0f, 3.25f };
    memcpy(input->data, in_data, sizeof(in_data));

    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    if (fe_runtime_run(&rt, input, output) != FE_OK) {
        printf("Ferrite: run failed\n");
        fe_tensor_free(input);
        fe_tensor_free(output);
        return;
    }

    float *o = (float *)output->data;
    printf("Ferrite ESP32 smoke: relu output = [%f, %f] (expect 0.000, 3.250)\n",
           o[0], o[1]);

    fe_tensor_free(input);
    fe_tensor_free(output);
}
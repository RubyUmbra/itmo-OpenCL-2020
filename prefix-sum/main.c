#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

#define TILE_SIZE 4096

struct state {
    size_t N;

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem mem_a_r;
    cl_mem mem_b_rw;
};

void release_state(struct state* st) {
    if (st->device) clReleaseDevice(st->device);
    if (st->context) clReleaseContext(st->context);
    if (st->queue) clReleaseCommandQueue(st->queue);
    if (st->program) clReleaseProgram(st->program);
    if (st->kernel) clReleaseKernel(st->kernel);

    if (st->mem_a_r) clReleaseMemObject(st->mem_a_r);
    if (st->mem_b_rw) clReleaseMemObject(st->mem_b_rw);

    free(st);
}

cl_int choose_device(struct state* st) {
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id* platforms;

    if ((err = clGetPlatformIDs(0, NULL, &num_platforms))) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        return err;
    }

    if ((platforms = calloc(num_platforms, sizeof(cl_platform_id))) == NULL) {
        return EXIT_FAILURE;
    }

    if ((err = clGetPlatformIDs(num_platforms, platforms, &num_platforms))) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        free(platforms);
        return err;
    }

    cl_uint num_devices = 0;
    size_t max_devices = 16;
    cl_device_id devices[max_devices];

    for (size_t i = 0; i < num_platforms; ++i) {
        if ((err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, max_devices, devices, &num_devices))) continue;
        for (size_t j = 0; j < num_devices; ++j) {
            if (st->device == NULL) {
                st->device = devices[j];
            } else {
                clReleaseDevice(devices[j]);
            }
        }
        if (st->device != NULL) { break; }
    }

    if (st->device == NULL) {
        for (size_t i = 0; i < num_platforms; ++i) {
            if ((err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, max_devices, devices, &num_devices))) continue;
            for (size_t j = 0; j < num_devices; ++j) {
                if (st->device == NULL) {
                    st->device = devices[j];
                } else {
                    clReleaseDevice(devices[j]);
                }
            }
            if (st->device != NULL) { break; }
        }
    }

    if (st->device == NULL) {
        fprintf(stderr, "No suitable devices.\n");
        free(platforms);
        return err;
    }

    char name[256];
    size_t ret_sz;
    clGetDeviceInfo(st->device, CL_DEVICE_NAME, 255, name, &ret_sz);
    name[ret_sz] = '\0';
    printf("Device: %s\n", name);

    st->context = clCreateContext(0, 1, &st->device, 0, 0, &err);
    if (err) {
        fprintf(stderr, "clCreateContext: %d\n", err);
        release_state(st);
        return err;
    }

    st->queue = clCreateCommandQueue(st->context, st->device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err) {
        fprintf(stderr, "clCreateCommandQueue: %d\n", err);
        release_state(st);
        return err;
    }

    return EXIT_SUCCESS;
}

cl_int set_program(struct state* st) {
    char const* file_data[0];
    size_t lengths[0];
    cl_int res = 0;

    memset(file_data, 0, sizeof(char const*) * 1);

    FILE* f = fopen("prefix-sum.cl", "r");
    if (f != NULL) {
        size_t const max_size = 10000;
        char* code = malloc(max_size);
        if (code != NULL) {
            size_t code_len = fread(code, 1, max_size - 1, f);
            code[code_len] = '\0';
            lengths[0] = code_len;
            fclose(f);
            file_data[0] = code;
            if (file_data[0] != NULL) {
                st->program = clCreateProgramWithSource(st->context, 1, file_data, lengths, &res);
                if (res == 0) {
                    if ((res = clBuildProgram(st->program, 1, &st->device, "", 0, 0))) {
                        fprintf(stderr, "Kernel compilation failed\n");
                        size_t log_len = 0;
                        cl_int save_err = res;
                        char* build_log;
                        if ((res = clGetProgramBuildInfo(st->program, st->device, CL_PROGRAM_BUILD_LOG, 0, 0,
                                                         &log_len)) == 0) {
                            build_log = malloc(log_len);
                            if ((res = clGetProgramBuildInfo(st->program, st->device, CL_PROGRAM_BUILD_LOG, log_len,
                                                             build_log, &log_len))) {
                                fprintf(stderr, "clGetProgramBuildInfo: %d", res);
                                free(build_log);
                            } else {
                                fprintf(stderr, "Kernel compilation log:\n%s\n", build_log);
                                res = save_err;
                            }
                        } else fprintf(stderr, "clGetProgramBuildInfo: %d\n", res);
                    }
                } else fprintf(stderr, "clCreateProgramWithSource: %d\n", res);
            } else res = -1;
        } else {
            fclose(f);
            res = -1;
        }
    } else {
        perror("Unable to open source file.");
        res = -1;
    }
    free((void*) file_data[0]);
    return res;
}

cl_int set_kernel(struct state* st) {
    cl_int res = 0;

    size_t N = st->N;

    st->kernel = clCreateKernel(st->program, "prefix-sum", &res);
    if (res != 0) {
        fprintf(stderr, "clCreateKernel: %d\n", res);
        return res;
    }

    st->mem_a_r = clCreateBuffer(st->context, CL_MEM_READ_ONLY, N * sizeof(float), 0, &res);
    if (res == 0) {
        st->mem_b_rw = clCreateBuffer(st->context, CL_MEM_READ_WRITE, N * sizeof(float), 0, &res);
        if (res == 0) {
            clSetKernelArg(st->kernel, 0, sizeof(cl_uint), &st->N);
            clSetKernelArg(st->kernel, 1, sizeof(cl_mem), &st->mem_a_r);
            clSetKernelArg(st->kernel, 2, sizeof(cl_mem), &st->mem_b_rw);

            return EXIT_SUCCESS;
        } else fprintf(stderr, "clCreateBuffer: %d\n", res);

        clReleaseMemObject(st->mem_a_r);
        st->mem_a_r = NULL;
    } else fprintf(stderr, "clCreateBuffer: %d\n", res);

    clReleaseKernel(st->kernel);
    st->kernel = NULL;
    return res;
}

struct state* get_state(size_t N) {
    struct state* context = calloc(1, sizeof(struct state));
    if (context == NULL) {
        return NULL;
    }

    context->N = N;

    if (!choose_device(context)) {
        if (!set_program(context)) {
            if (!set_kernel(context)) {
                return context;
            }
        }
    }

    release_state(context);
    return NULL;
}

void prefix_sum_ocl(const float* a, float* b, uint N,
                    cl_ulong* time_start, cl_ulong* time_end) {
    struct state* st;
    cl_int err;

    printf("Eval using OpenCL.\n");

    if ((st = get_state(N)) == NULL) {
        fprintf(stderr, "Unable to startup.\n");
        return;
    }

    if (!(err = clEnqueueWriteBuffer(st->queue, st->mem_a_r, true, 0, N * sizeof(float), a, 0, 0, 0))) {
        size_t work_size[] = {N};
        size_t local_group_size[] = {TILE_SIZE};
        cl_event event;
        if (!(err = clEnqueueNDRangeKernel(st->queue, st->kernel, 1, NULL, work_size, local_group_size, 0, 0,
                                           &event))) {
            clEnqueueReadBuffer(st->queue, st->mem_b_rw, true, 0, N * sizeof(float), b, 0, 0, 0);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), time_start, 0);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), time_end, 0);
        } else fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", err);
    } else fprintf(stderr, "clEnqueueWriteBuffer: %d\n", err);

    release_state(st);
}

void prefix_sum_raw(const float* a, float* b, uint N) {
    printf("Eval prefix sum raw.\n");
    if (N != 0) b[0] = a[0];
    for (size_t i = 1; i < N; ++i) {
        b[i] = b[i - 1] + a[i];
    }
}

void check_result(const float* b, const float* c, const uint N) {
    printf("Checking results.\n");
    for (size_t i = 0; i < N; ++i) {
        float o1 = b[i];
        float o2 = c[i];
        float diff = fabsf(o1 - o2);
        if (o1 != o2) {
            if (o1 == 0 || o2 == 0 || (fabsf(o1) + fabsf(o2) < FLT_MIN)) {
                assert(diff < 0.001);
            } else {
                assert((diff / (fabsf(o1) + fabsf(o2))) < 0.001);
            }
        }
    }
}

void gen_array(float* a, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        a[i] = (float) rand() / (float) RAND_MAX;
    }
}

/// N = TILE_SIZE
int main() {
    const size_t N = TILE_SIZE;
    float* a;
    float* b;
    float* c;

    a = calloc(N, sizeof(float));
    b = calloc(N, sizeof(float));
    c = calloc(N, sizeof(float));

    if (a != NULL && b != NULL && c != NULL) {
        gen_array(a, N);

        cl_ulong time_start = 0;
        cl_ulong time_end = 0;

        prefix_sum_raw(a, b, N);
        prefix_sum_ocl(a, c, N, &time_start, &time_end);
        check_result(b, c, N);

        long double time_diff = time_end - time_start;
        printf("%.2Lf ms\n", time_diff / 1e6);
    }

    if (c != NULL) free(c);
    if (b != NULL) free(b);
    if (a != NULL) free(a);
    return EXIT_SUCCESS;
}

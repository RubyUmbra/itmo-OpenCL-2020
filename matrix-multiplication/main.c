#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

#define TILE_SIZE 32
#define ELEMENTS_PER_THREAD 4

struct state {
    size_t N;
    size_t K;
    size_t M;

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem mem_a_r;
    cl_mem mem_b_r;
    cl_mem mem_c_rw;
};

void release_state(struct state* st) {
    if (st->device) clReleaseDevice(st->device);
    if (st->context) clReleaseContext(st->context);
    if (st->queue) clReleaseCommandQueue(st->queue);
    if (st->program) clReleaseProgram(st->program);
    if (st->kernel) clReleaseKernel(st->kernel);

    if (st->mem_a_r) clReleaseMemObject(st->mem_a_r);
    if (st->mem_b_r) clReleaseMemObject(st->mem_b_r);
    if (st->mem_c_rw) clReleaseMemObject(st->mem_c_rw);

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

    platforms = calloc(num_platforms, sizeof(cl_platform_id));
    if (platforms == NULL) {
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

    FILE* f = fopen("matrix_mul.cl", "r");
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
    size_t K = st->K;
    size_t M = st->M;

    st->kernel = clCreateKernel(st->program, "matrix_mul", &res);
    if (res != 0) {
        fprintf(stderr, "clCreateKernel: %d\n", res);
        return res;
    }

    st->mem_a_r = clCreateBuffer(st->context, CL_MEM_READ_ONLY, M * K * sizeof(float), 0, &res);
    if (res == 0) {
        st->mem_b_r = clCreateBuffer(st->context, CL_MEM_READ_ONLY, K * N * sizeof(float), 0, &res);
        if (res == 0) {
            st->mem_c_rw = clCreateBuffer(st->context, CL_MEM_READ_WRITE, M * N * sizeof(float), 0, &res);
            if (res == 0) {
                clSetKernelArg(st->kernel, 0, sizeof(cl_mem), &st->mem_a_r);
                clSetKernelArg(st->kernel, 1, sizeof(cl_mem), &st->mem_b_r);
                clSetKernelArg(st->kernel, 2, sizeof(cl_mem), &st->mem_c_rw);
                clSetKernelArg(st->kernel, 3, sizeof(cl_uint), &st->N);
                clSetKernelArg(st->kernel, 4, sizeof(cl_uint), &st->K);
                clSetKernelArg(st->kernel, 5, sizeof(cl_uint), &st->M);

                return EXIT_SUCCESS;
            } else fprintf(stderr, "clCreateBuffer: %d\n", res);

            clReleaseMemObject(st->mem_b_r);
            st->mem_b_r = NULL;
        } else fprintf(stderr, "clCreateBuffer: %d\n", res);

        clReleaseMemObject(st->mem_a_r);
        st->mem_a_r = NULL;
    } else fprintf(stderr, "clCreateBuffer: %d\n", res);

    clReleaseKernel(st->kernel);
    st->kernel = NULL;
    return res;
}

struct state* get_state(size_t N, size_t K, size_t M) {
    struct state* context = calloc(1, sizeof(struct state));
    if (context == NULL) {
        return NULL;
    }

    context->N = N;
    context->K = K;
    context->M = M;

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

void matrix_mul_ocl(const float* a, const float* b, float* c, uint N, uint K, uint M,
                    cl_ulong* time_start, cl_ulong* time_end) {
    struct state* st;
    cl_int err;

    printf("Eval using OpenCL.\n");

    if ((st = get_state(N, K, M)) == NULL) {
        fprintf(stderr, "Unable to startup.\n");
        return;
    }

    if (!(err = clEnqueueWriteBuffer(st->queue, st->mem_a_r, true, 0, M * K * sizeof(float), a, 0, 0, 0))) {
        if (!(err = clEnqueueWriteBuffer(st->queue, st->mem_b_r, true, 0, K * N * sizeof(float), b, 0, 0, 0))) {
            size_t work_size[] = {N, M / ELEMENTS_PER_THREAD};
            size_t local_group_size[] = {TILE_SIZE, TILE_SIZE / ELEMENTS_PER_THREAD};
            cl_event event;
            if (!(err = clEnqueueNDRangeKernel(st->queue, st->kernel, 2, NULL, work_size, local_group_size, 0, 0,
                                               &event))) {
                clEnqueueReadBuffer(st->queue, st->mem_c_rw, true, 0, M * N * sizeof(float), c, 0, 0, 0);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), time_start, 0);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), time_end, 0);
            } else fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", err);
        } else fprintf(stderr, "clEnqueueWriteBuffer: %d\n", err);
    } else fprintf(stderr, "clEnqueueWriteBuffer: %d\n", err);

    release_state(st);
}

void matrix_mul_omp(const float* a, const float* b, float* c, uint N, uint K, uint M) {
    printf("Eval using OpenMP.\n");

#pragma omp parallel for default(shared) schedule(dynamic)
    for (size_t y = 0; y < M; ++y) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t x = 0; x < N; ++x) {
                c[y * N + x] += a[y * K + i] * b[i * N + x];
            }
        }
    }
}

void check_result(const float* c, const float* s, const uint N, const uint M) {
    printf("Checking results.\n");
    for (size_t i = 0; i < M * N; ++i) {
        assert(fabsf(s[i] - c[i]) < 0.00001);
    }
}

void gen_matrix(float* a, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        a[i] = (float) rand() / (float) RAND_MAX;
    }
}

/// N, N, K must be divisible by tile size
int main() {
    const size_t N = 2048;
    const size_t K = 512;
    const size_t M = 1024;
    float* a;
    float* b;
    float* c;
    float* s;

    a = calloc(M * K, sizeof(float));
    b = calloc(K * N, sizeof(float));
    c = calloc(M * N, sizeof(float));
    s = calloc(M * N, sizeof(float));

    if (a != NULL && b != NULL && c != NULL && s != NULL) {
        gen_matrix(a, M * K);
        gen_matrix(b, K * N);

        cl_ulong time_start = 0;
        cl_ulong time_end = 0;

        matrix_mul_omp(a, b, s, N, K, M);
        matrix_mul_ocl(a, b, c, N, K, M, &time_start, &time_end);
        check_result(c, s, N, M);

        long double time_diff = time_end - time_start;
        printf("%.2Lf ms | %.2Lf GFLOPS\n", time_diff / 1e6, (M * K * N * 2) / time_diff);
    }

    if (s != NULL) free(s);
    if (c != NULL) free(c);
    if (b != NULL) free(b);
    if (a != NULL) free(a);
    return EXIT_SUCCESS;
}

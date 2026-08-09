/* gpu_kernel.c -- element-wise kernels compiled at run time (GPU::Kernel).
 *
 * The Ruby side (mrblib/gpu_kernel.rb) traces a block into GLSL source. This
 * file takes that source, runs it through glslangValidator, builds a compute
 * pipeline from the SPIR-V, and dispatches it.
 *
 * Everything here rides on mruby-gpu-narray: the Vulkan context (g_ctx), the
 * descriptor-set layouts, and the buffer helpers all come from its
 * gpu_internal.h. A traced kernel is one input buffer and one output buffer,
 * which is exactly LAYOUT_2BUF, so no new descriptor plumbing is needed.
 *
 * The dispatch below duplicates the body of mruby-gpu-narray's
 * dispatch_compute(). Upstream's version is keyed to its PipeId enum and so
 * cannot bind a pipeline we built ourselves; when it grows a pipeline-taking
 * variant, this copy should go away.
 *
 * We shell out to glslangValidator rather than linking glslang, so the link
 * dependency stays libvulkan only -- and the compiler is already required to
 * build mruby-gpu-narray's own shaders.
 */
#include "gpu_internal.h"
#include <mruby/error.h>
#include <unistd.h>

typedef struct {
  VkPipeline pipeline;
} GpuKernel;

static void kernel_free(mrb_state *mrb, void *p) {
  GpuKernel *k = (GpuKernel *)p;
  if (!k) return;
  if (k->pipeline != VK_NULL_HANDLE && g_ctx.initialized) {
    vkDestroyPipeline(g_ctx.device, k->pipeline, NULL);
  }
  mrb_free(mrb, k);
}

static const struct mrb_data_type gpu_kernel_type = {
  "GPU::Kernel", kernel_free
};

/* ---- file helpers ---- */

static uint8_t *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len < 0) { fclose(f); return NULL; }
  uint8_t *buf = (uint8_t *)malloc((size_t)len + 1);
  if (buf) {
    size_t got = fread(buf, 1, (size_t)len, f);
    buf[got] = '\0';
    if (size) *size = got;
  }
  fclose(f);
  return buf;
}

/* ---- Vulkan ---- */

static VkPipeline pipeline_from_spv(const char *path) {
  size_t spv_size = 0;
  uint8_t *spv = read_file(path, &spv_size);
  if (!spv) return VK_NULL_HANDLE;

  VkShaderModuleCreateInfo sm_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spv_size,
    .pCode = (uint32_t *)spv
  };
  VkShaderModule shader;
  if (vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader) != VK_SUCCESS) {
    free(spv);
    return VK_NULL_HANDLE;
  }
  free(spv);

  VkComputePipelineCreateInfo cp_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main"
    },
    .layout = g_ctx.pipe_layouts[LAYOUT_2BUF]
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info,
                               NULL, &pipeline) != VK_SUCCESS) {
    pipeline = VK_NULL_HANDLE;
  }
  vkDestroyShaderModule(g_ctx.device, shader, NULL);
  return pipeline;
}

/* One input buffer, one output buffer, { uint n } push constant. */
static void dispatch_kernel(VkPipeline pipeline,
                            VkBuffer *buffers, VkDeviceSize *sizes,
                            uint32_t n, uint32_t groups) {
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layouts[LAYOUT_2BUF]
  };
  VkDescriptorSet desc_set;
  vkAllocateDescriptorSets(g_ctx.device, &dsai, &desc_set);

  VkDescriptorBufferInfo buf_infos[2];
  VkWriteDescriptorSet writes[2];
  for (int i = 0; i < 2; i++) {
    buf_infos[i] = (VkDescriptorBufferInfo){buffers[i], 0, sizes[i]};
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buf_infos[i]
    };
  }
  vkUpdateDescriptorSets(g_ctx.device, 2, writes, 0, NULL);

  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_ctx.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(g_ctx.device, &cbai, &cmd);

  VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  vkBeginCommandBuffer(cmd, &begin);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layouts[LAYOUT_2BUF], 0, 1, &desc_set, 0, NULL);
  vkCmdPushConstants(cmd, g_ctx.pipe_layouts[LAYOUT_2BUF],
    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n);
  vkCmdDispatch(cmd, groups, 1, 1);
  vkEndCommandBuffer(cmd);

  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(g_ctx.device, &fi, NULL, &fence);

  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd
  };
  vkQueueSubmit(g_ctx.queue, 1, &si, fence);
  vkWaitForFences(g_ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(g_ctx.device, fence, NULL);
  vkFreeCommandBuffers(g_ctx.device, g_ctx.cmd_pool, 1, &cmd);
  vkFreeDescriptorSets(g_ctx.device, g_ctx.desc_pool, 1, &desc_set);
}

/* ---- mruby surface ---- */

/* GPU::Kernel.compile(glsl_source) -> GPU::Kernel
 *
 * Raises GPU::CompileError carrying glslang's own output, so a bad block reads
 * like a Ruby error rather than a silent failure.
 *
 * Vulkan must already be up. mruby-gpu-narray initializes lazily on its first
 * operation and keeps that entry point private, so the Ruby side touches
 * GPU.info before calling here.
 */
static mrb_value kernel_s_compile(mrb_state *mrb, mrb_value self) {
  const char *src;
  mrb_int src_len;
  mrb_get_args(mrb, "s", &src, &src_len);

  struct RClass *gpu = mrb_module_get(mrb, "GPU");
  struct RClass *cerr = mrb_class_get_under(mrb, gpu, "CompileError");

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR,
      "no Vulkan device yet -- create a GPU::NArray (or call GPU.info) first");
  }

  const char *tmpdir = getenv("TMPDIR");
  if (!tmpdir || !*tmpdir) tmpdir = "/tmp";

  static unsigned long seq = 0;
  unsigned long id = seq++;
  char comp_path[1024], spv_path[1024], log_path[1024], cmd[4096];
  snprintf(comp_path, sizeof(comp_path), "%s/mruby-gpu-kernel-%ld-%lu.comp",
           tmpdir, (long)getpid(), id);
  snprintf(spv_path, sizeof(spv_path), "%s/mruby-gpu-kernel-%ld-%lu.spv",
           tmpdir, (long)getpid(), id);
  snprintf(log_path, sizeof(log_path), "%s/mruby-gpu-kernel-%ld-%lu.log",
           tmpdir, (long)getpid(), id);

  FILE *f = fopen(comp_path, "wb");
  if (!f) mrb_raisef(mrb, E_RUNTIME_ERROR, "cannot write %s", comp_path);
  fwrite(src, 1, (size_t)src_len, f);
  fclose(f);

  const char *glslang = getenv("GLSLANG");
  if (!glslang || !*glslang) glslang = "glslangValidator";

  snprintf(cmd, sizeof(cmd), "\"%s\" -V \"%s\" -o \"%s\" > \"%s\" 2>&1",
           glslang, comp_path, spv_path, log_path);
  int rc = system(cmd);

  if (rc != 0) {
    uint8_t *log = read_file(log_path, NULL);
    mrb_value msg = mrb_str_new_cstr(mrb, log ? (const char *)log
      : "glslangValidator failed (is it on PATH? override with GLSLANG=)");
    free(log);
    remove(comp_path);
    remove(log_path);
    remove(spv_path);
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, msg));
  }

  VkPipeline pipeline = pipeline_from_spv(spv_path);
  remove(comp_path);
  remove(log_path);
  remove(spv_path);

  if (pipeline == VK_NULL_HANDLE) {
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, mrb_str_new_cstr(mrb,
      "SPIR-V compiled but the compute pipeline could not be created")));
  }

  GpuKernel *k = (GpuKernel *)mrb_malloc(mrb, sizeof(GpuKernel));
  k->pipeline = pipeline;

  struct RObject *obj = (struct RObject *)mrb_obj_alloc(mrb, MRB_TT_CDATA,
                                                        mrb_class_ptr(self));
  mrb_value kernel = mrb_obj_value(obj);
  DATA_PTR(kernel) = k;
  DATA_TYPE(kernel) = &gpu_kernel_type;
  mrb_iv_set(mrb, kernel, mrb_intern_lit(mrb, "@glsl"),
             mrb_str_new(mrb, src, src_len));
  return kernel;
}

/* kernel.call(narray) -> narray -- run the kernel over every element. */
static mrb_value kernel_call(mrb_state *mrb, mrb_value self) {
  mrb_value arg;
  mrb_get_args(mrb, "o", &arg);

  GpuKernel *k = (GpuKernel *)DATA_GET_PTR(mrb, self, &gpu_kernel_type, GpuKernel);

  if (mrb_type(arg) != MRB_TT_CDATA || DATA_TYPE(arg) != &gpu_buffer_type) {
    mrb_raisef(mrb, E_TYPE_ERROR, "GPU::Kernel#call expects a GPU::NArray, got %s",
               mrb_obj_classname(mrb, arg));
  }
  GpuBuffer *a = (GpuBuffer *)DATA_PTR(arg);
  GpuBuffer *b = create_buffer(mrb, a->n);

  VkBuffer bufs[2]      = {a->buffer, b->buffer};
  VkDeviceSize sizes[2] = {a->bytes,  b->bytes};
  dispatch_kernel(k->pipeline, bufs, sizes, a->n, (a->n + 255) / 256);

  return wrap_buffer(mrb, mrb_obj_class(mrb, arg), b);
}

void mrb_mruby_gpu_kernel_gem_init(mrb_state *mrb) {
  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  mrb_define_class_under(mrb, gpu, "CompileError", mrb_class_get(mrb, "StandardError"));

  struct RClass *kernel = mrb_define_class_under(mrb, gpu, "Kernel", mrb->object_class);
  MRB_SET_INSTANCE_TT(kernel, MRB_TT_CDATA);
  mrb_define_class_method(mrb, kernel, "compile", kernel_s_compile, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, kernel, "call", kernel_call, MRB_ARGS_REQ(1));
}

void mrb_mruby_gpu_kernel_gem_final(mrb_state *mrb) {
  (void)mrb;
}

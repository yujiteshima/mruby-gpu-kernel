MRuby::Gem::Specification.new('mruby-gpu-kernel') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.version = '0.2.0'
  spec.summary = 'Trace a Ruby block into a Vulkan compute shader for mruby-gpu-narray'

  # Resolved from mgem-list when no source is given (lib/mruby/gem.rb:
  # default_gem_params falls back to { mgem: <name> }).
  spec.add_dependency 'mruby-gpu-narray'

  # The GPU is reached only through mruby-gpu-runtime's C API (gpu_runtime.h),
  # so this gem names it as a dependency of its own, although
  # mruby-gpu-narray already brings it. The include path and the Vulkan loader,
  # the one link dependency, come with it. It is not on mgem-list yet; name the
  # branch, because mruby clones "master" when none is given and the repository
  # has only "main".
  spec.add_dependency 'mruby-gpu-runtime', github: 'yujiteshima/mruby-gpu-runtime', branch: 'main'
end

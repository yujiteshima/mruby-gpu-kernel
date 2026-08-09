# Reference build_config.rb for mruby-gpu-kernel.
#
# Only this gem is listed: mruby-gpu-narray is pulled in automatically as a
# dependency, resolved through mgem-list (mruby's official gem catalog).
#
# Usage:
#   cd /path/to/mruby
#   MRUBY_CONFIG=/path/to/mruby-gpu-kernel/build_config.reference.rb rake
#   ./build/host/bin/mruby /path/to/mruby-gpu-kernel/test/kernel_test.rb
#
# To develop against a local checkout of the base gem instead of the published
# one, add it before this gem:
#   conf.gem File.expand_path('~/dev/personal/mruby-gpu-narray')

MRuby::Build.new do |conf|
  toolchain :clang        # Pi: :gcc (or just `toolchain` to auto-detect)

  conf.gembox 'default'   # provides the mruby / mirb / mrbc executables

  conf.gem File.expand_path('~/dev/personal/mruby-gpu-kernel')

  # --- macOS only (Homebrew Vulkan). Remove these two lines on the Pi. ---
  conf.cc.include_paths     << '/opt/homebrew/include'
  conf.linker.library_paths << '/opt/homebrew/lib'
  # -----------------------------------------------------------------------

  conf.enable_test
end

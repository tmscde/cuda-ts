{
  "targets": [
    {
      "target_name": "cuda",
      "sources": [
        "src/native/shared.cpp",
        "src/native/addon.cpp",
        "src/native/device.cpp",
        "src/native/context.cpp",
        "src/native/module.cpp",
        "src/native/memory.cpp",
        "src/native/function.cpp",
        "src/native/compiler.cpp",
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": ["<!(node -p \"require('node-addon-api').gyp\")"],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "conditions": [
        [ 'OS=="mac"', {
          "libraries": ["-framework CUDA"],
          "include_dirs": ['/usr/local/include'],
          "library_dirs": ['/usr/local/lib'],
          "cflags+": ['-fvisibility=hidden'],
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.7",
            "GCC_SYMBOLS_PRIVATE_EXTERN": 'YES', # -fvisibility=hidden
          },
        }],
        [ 'OS=="linux"', {
          'libraries': ['-lcuda'],
          'include_dirs': ['/usr/local/include'],
          'library_dirs': ['/usr/local/lib']
        }],
        [ 'OS=="win"', {
          "msvs_settings": {
            "VCCLCompilerTool": { "ExceptionHandling": 1 },
          },
          "conditions": [
            ['target_arch=="x64"',
              {
                'variables': { 'arch': 'x64' }
              }, {
                'variables': { 'arch': 'Win32' }
              }
            ],
          ],
          'variables': {
            'cuda_root%': '$(CUDA_PATH)'
          },
          'libraries': [
            '-l<(cuda_root)/lib/<(arch)/cuda.lib',
            '-l<(cuda_root)/lib/<(arch)/nvrtc.lib',
          ],
          "include_dirs": [
            "<(cuda_root)/include",
          ],
        }, {
          "include_dirs": [
            "/usr/local/cuda-5.0/include",
            "/usr/local/cuda/include"
          ],
        }]
      ]
    }
  ]
}

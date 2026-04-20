module.exports = {
  run: [
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/facebookresearch/ai4animationpy app"
      }
    },
    {
      when: "{{exists('app/.git')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    {
      when: "{{exists('app/env')}}",
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      when: "{{exists('app/.conda')}}",
      method: "fs.rm",
      params: {
        path: "app/.conda"
      }
    },
    {
      when: "{{exists('app/app/env')}}",
      method: "fs.rm",
      params: {
        path: "app/app/env"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        path: "app",
        message: [
          "uv pip install torch torchvision torchaudio numpy scipy matplotlib scikit-learn einops pygltflib==1.16.5 pyscreenrec==0.6 tqdm pyyaml onnx==1.19.1 raylib",
          "uv pip install fastapi \"uvicorn[standard]\"",
          "uv pip install -e . --no-deps"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia' && (platform === 'linux' || platform === 'win32')}}",
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        path: "app",
        message: "uv pip install onnxruntime-gpu"
      }
    },
    {
      when: "{{!(gpu === 'nvidia' && (platform === 'linux' || platform === 'win32'))}}",
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        path: "app",
        message: "uv pip install onnxruntime"
      }
    },
    {
      method: "notify",
      params: {
        html: "AI4AnimationPy updated."
      }
    }
  ]
}

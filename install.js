module.exports = {
  "run": [
    // Install PyTorch and related packages (includes venv creation)
    {
      "method": "script.start",
      "params": {
        "uri": "torch.js",
        "params": {
          "venv_python": "3.12",
          "venv": "env",
          "path": "app",
        }
      }
    },
    // Core Python dependencies
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": [
          "uv pip install liquid-audio",
          "uv pip install liquid-audio[demo]",
          "uv pip install -U triton-windows" 
        ]
      }
    },
    // Additional Python dependencies
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": [
          "uv pip install gradio devicetorch",
          "uv pip install librosa",
          "uv pip install soundfile",
          "uv pip install accelerate"
        ]
      }
    },
    {
      "method": "shell.run",
      "params": {
        "venv": "env",
        "path": "app",
        "message": [
          "uv pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe pillow"
        ]
      }
    },
    // System dependencies (platform-specific)
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "message": [
          "brew install ffmpeg libsndfile cmake gcc || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gcc) installation failed. Please install manually with: brew install ffmpeg libsndfile cmake gcc'"
        ]
      }
    },
    {
      "when": "{{platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "message": [
          "sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1-dev cmake gfortran || sudo yum install -y ffmpeg libsndfile-devel cmake gcc-gfortran || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gfortran) installation failed. Please install manually with: sudo apt-get install ffmpeg libsndfile1-dev cmake gfortran or sudo yum install ffmpeg libsndfile-devel cmake gcc-gfortran'"
        ]
      }
    },
    {
      "when": "{{platform === 'win32'}}",
      "method": "shell.run",
      "params": {
        "message": [
          "conda install ffmpeg -c conda-forge --yes || echo 'FFmpeg installation failed. Please install manually with: conda install ffmpeg'"
        ]
      }
    },
    // Notify user
    {
      "method": "notify",
      "params": {
        "html": "Installation complete. Click the 'start' tab to launch Liquid-Audio!"
      }
    }
  ]
};





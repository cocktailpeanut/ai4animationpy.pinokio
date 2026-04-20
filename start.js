module.exports = async (kernel) => {
  const PORT = await kernel.port()
  return {
    daemon: true,
    run: [{
      method: "shell.run",
      params: {
        path: ".",
        venv: "app/env",
        venv_python: "3.12",
        env: {
          HOST: "127.0.0.1",
          PORT: `${PORT}`
        },
        message: "python webdemo/run.py",
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    }, {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }]
  }
}

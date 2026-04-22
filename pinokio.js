module.exports = {
  version: "7.0",
  title: "AI4AnimationPy",
  icon: "icon.png",
  description: "Minimal local web playground for AI4AnimationPy character and motion demos",
  menu: (kernel, info) => {
    let installed = info.exists("app") && info.exists("app/env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }]
    }

    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }]
    }

    if (running.start) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web App",
          href: local.url
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js"
        }]
      }
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Starting web app",
        href: "start.js"
      }]
    }

    if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Updating",
        href: "update.js"
      }]
    }

    if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-circle-xmark",
        text: "Resetting",
        href: "reset.js"
      }]
    }

    return [{
      default: true,
      icon: "fa-solid fa-globe",
      text: "Start Web App",
      href: "start.js"
    }, {
      icon: "fa-solid fa-plug",
      text: "Update",
      href: "update.js"
    }, {
      icon: "fa-solid fa-circle-xmark",
      text: "Reset",
      href: "reset.js",
      confirm: "This removes the cloned app and installed environment folder. Continue?"
    }]
  }
}

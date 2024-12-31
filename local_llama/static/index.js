const ask_btn = document.getElementById("btn_1");
const ask_ws_btn = document.getElementById("btn_2");
const ask_sse_btn = document.getElementById("btn_3");

const text_ele = document.getElementById("text");
const response_box = document.getElementById("textBox");
const popover_ele = document.getElementById("my-popover");
ask_btn.addEventListener("click", async (event) => {
  event.preventDefault();
  if (text_ele.value == "") {
    popover_ele.showPopover();
    return;
  }
  response_box.innerText = "";
  const text = text_ele.value;
  text_ele.value = "";
  const response_pro = fetch(`/search?query=${text}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });
  const res = await response_pro;
  if (!res.ok) {
    console.log("Error");
    return;
  }
  response_box.innerText = _.trim(await res.text(), '"');
});

ask_ws_btn.addEventListener("click", async (event) => {
  event.preventDefault();
  if (text_ele.value == "") {
    popover_ele.showPopover();
    return;
  }
  response_box.innerText = "";
  const text = text_ele.value;
  text_ele.value = "";
  const ws = new WebSocket("ws://localhost:8000/ws");
  ws.onmessage = function (event) {
    response_box.innerText += event.data;
  };

  ws.onopen = function () {
    ws.send(text);
  };
});

ask_sse_btn.addEventListener("click", async (event) => {
  event.preventDefault();
  if (text_ele.value == "") {
    popover_ele.showPopover();
    return;
  }
  response_box.innerText = "";
  const text = text_ele.value;
  text_ele.value = "";
  const eventSource = new EventSource(`/sse?query=${text}`);
  eventSource.onmessage = function (event) {
    console.log(event);
    let trimmed_data = _.trim(event.data, '"');
    if (trimmed_data == "") {
      eventSource.close();
    }
    response_box.innerText += trimmed_data;
  };
  eventSource.onerror = function (event) {
    console.log("Error", event);
  };
  eventSource.onopen = function (event) {
    //这也能执行
    console.log("EventSource connection opened");
  };
  eventSource.addEventListener("close", function (event) {
    console.log(event);
    eventSource.close();
  });
});

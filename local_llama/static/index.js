const ask_btn = document.getElementById("btn");

const text_ele = document.getElementById("text");
const response_box = document.getElementById("textBox");
ask_btn.addEventListener("click", async (event) => {
  event.preventDefault();
  const text = text_ele.value;
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

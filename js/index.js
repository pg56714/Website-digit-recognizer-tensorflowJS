async function init() {
  model = await tf.loadLayersModel("./models/model.json");
  console.log("load model...");
}

function submit() {
  // 因為可上傳多筆照片，但模型只能吃一筆，故只呼叫第一筆資料
  const selectFile = document.getElementById("input").files[0];
  console.log(selectFile);
  // 檔案載入
  let reader = new FileReader();
  reader.onload = (e) => {
    // 建立一個img標籤，ex:<img>
    let img = document.createElement("img");
    // 圖片來源
    img.src = e.target.result;
    img.width = 144;
    img.height = 144;
    img.onload = () => {
      const showImage = document.getElementById("showImage");
      // 先清空
      showImage.innerHTML = "";
      // 再加入圖片
      showImage.appendChild(img);
      predict(img);
    };
  };
  // 讀取檔案
  reader.readAsDataURL(selectFile);
}

function findMaxIndex(result) {
  const arr = Array.from(result);
  let maxIndex = 0;
  let maxValue = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxValue) {
      maxIndex = i;
      maxValue = arr[i];
    }
  }
  return { predNum: maxIndex, prob: maxValue };
}

function predict(imgElement) {
  // 將 HTML <img> 轉換成轉換為矩陣 tensor，且只取單色，後面的 1 代表灰階，因為是黑白照片
  const tfImg = tf.browser.fromPixels(imgElement, 1);
  // 強制將圖片縮小到 28*28 像素
  const smalImg = tf.image.resizeBilinear(tfImg, [28, 28]);
  // 將 tensor 設為浮點型態，且將張量攤平至一為矩陣。此時 shape 為 [1, 784]
  let tensor = smalImg.reshape([1, -1]);
  // 將所有數值除以255
  tensor = tensor.div(tf.scalar(255));
  // 預測
  const pred = model.predict(tensor);
  // 取得預測結果
  const result = pred.dataSync();
  // 取得 one hot encoding 陣列中最大的索引
  const { predNum, prob } = findMaxIndex(result);
  console.log(predNum, prob);
  // 顯示預測結果
  document.getElementById("resultValue").innerHTML = predNum;
}

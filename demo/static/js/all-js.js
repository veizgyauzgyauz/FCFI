// Special thanks to:
//   - Zhanwei Xu
//   - Zhangliang Sun

const sliderValueLimit = 10

const fileInput = document.getElementById('imageUpload');
let img = new Image();

fileInput.addEventListener('change', (e) => {
const file = e.target.files[0];

if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        img.src = event.target.result;
    }
    reader.readAsDataURL(file);
}
});


const sliderInput = document.querySelectorAll("input")[1];
const sliderLabel = document.querySelectorAll("label")[1];

sliderInput.addEventListener("input", event => {
    const sliderValue = Number(sliderInput.value) / 100;
    sliderInput.style.setProperty("--thumb-rotate", `${sliderValue * 720}deg`);
    sliderLabel.innerHTML = Math.round(sliderValue * sliderValueLimit);
});



const pointType = document.getElementById('pointType');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const saveBtn = document.getElementById('saveBtn');
const sourceCanvas = document.getElementById('sourceCanvas');
const resultCanvas = document.getElementById('resultCanvas');
const sourceCtx = sourceCanvas.getContext('2d');
const resultCtx = resultCanvas.getContext('2d');

var foregroundPoints = [];
var backgroundPoints = [];

img.onload = () => {
    sourceCanvas.width = img.width;
    sourceCanvas.height = img.height;
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;
    sourceCtx.drawImage(img, 0, 0, img.width, img.height);
};

sourceCanvas.addEventListener('click', (e) => {
    var x = e.offsetX / sourceCanvas.clientWidth * sourceCanvas.width;
    var y = e.offsetY / sourceCanvas.clientHeight * sourceCanvas.height;
    
    if (pointType.value === 'foreground') {
        foregroundPoints.push({ x, y });
        sourceCtx.fillStyle = 'green';
    } else {
        backgroundPoints.push({ x, y });
        sourceCtx.fillStyle = 'red';
    }

    sourceCtx.beginPath();
    sourceCtx.arc(x, y, 5, 0, 2 * Math.PI);
    sourceCtx.fill();
});

sendBtn.addEventListener('click', async () => {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    const sliderValue = Number(sliderInput.value) / 100 * sliderValueLimit;

    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    tempCtx.drawImage(img, 0, 0, img.width, img.height);

    var url = '/submit_post_message';
    const data = {
        image: tempCanvas.toDataURL('image/jpeg'),
        foregroundPoints,
        backgroundPoints,
        sliderValue,
    };

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
        
    const result = await response.json();
    const resultImg = new Image();
    resultImg.src = "data:image/jpeg;base64," + result.output_image;
    resultImg.onload = () => {
        sourceCtx.drawImage(resultImg, 0, 0, resultImg.width, resultImg.height);

        sourceCtx.fillStyle = 'green';
        for (let i = 0; i < foregroundPoints.length; i++) {
            let x = foregroundPoints[i].x;
            let y = foregroundPoints[i].y;
            
            sourceCtx.beginPath();
            sourceCtx.arc(x, y, 5, 0, 2 * Math.PI);
            sourceCtx.fill();
        }
    
        sourceCtx.fillStyle = 'red';
        for (let i = 0; i < backgroundPoints.length; i++) {
            let x = backgroundPoints[i].x;
            let y = backgroundPoints[i].y;
            
            sourceCtx.beginPath();
            sourceCtx.arc(x, y, 5, 0, 2 * Math.PI);
            sourceCtx.fill();
        }
    }

    const resultMask = new Image();
    resultMask.src = "data:image/jpeg;base64," + result.output_mask;
    resultMask.onload = () => {
        resultCtx.drawImage(resultMask, 0, 0, resultMask.width, resultMask.height);
    }
});

clearBtn.addEventListener('click', async () => {
    sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    foregroundPoints = []
    backgroundPoints = []

    var url = '/clear_post_message';
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
});

saveBtn.addEventListener('click', () => {
    const url = resultCanvas.toDataURL('image/png');

    const downloadUrl = document.createElement('a');
    downloadUrl.setAttribute('href', url);

    downloadUrl.setAttribute('download', 'segmentation_mask.png');

    document.body.appendChild(downloadUrl);
    downloadUrl.click();
    document.body.removeChild(downloadUrl);
});

var colorPicker = document.getElementById("colorPicker");
var colorPickerBtn = document.getElementById("colorPickerBtn");

function showColorPicker() {
    if (colorPicker.style.display === "none") {
        colorPicker.style.display = "block";
    } else {
        colorPicker.style.display = "none";
    }
}

// https://www.npmjs.com/package/vue-color
// https://github.com/xiaokaike/vue-color
// https://github.com/xiaokaike/vue-color#readme
Vue.component("color-picker", VueColor.Sketch), new Vue({
  el: "#vue_sketch_picker",
  data: function() {
    return {
      colors: {
        rgba: {
            r: 128,
            g: 0,
            b: 0,
            a: 0.6,
        }
      }
    }
  },
  methods: {
    updateValue: async function(selectedColor) {
        console.log(selectedColor, selectedColor.rgba.r, selectedColor.rgba.g, selectedColor.rgba.b, selectedColor.rgba.a)
        
        const data = {
            color_r: selectedColor.rgba.r,
            color_g: selectedColor.rgba.g,
            color_b: selectedColor.rgba.b,
            color_a: selectedColor.rgba.a,
        };
        
        const response = await fetch('/change_color_post_message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        

        const result = await response.json();
        const resultImg = new Image();
        resultImg.src = "data:image/jpeg;base64," + result.output_image;
        resultImg.onload = () => {
            sourceCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
            sourceCtx.drawImage(resultImg, 0, 0, resultImg.width, resultImg.height);

            sourceCtx.fillStyle = 'green';
            for (let i = 0; i < foregroundPoints.length; i++) {
                let x = foregroundPoints[i].x;
                let y = foregroundPoints[i].y;
                
                sourceCtx.beginPath();
                sourceCtx.arc(x, y, 5, 0, 2 * Math.PI);
                sourceCtx.fill();
            }
        
            sourceCtx.fillStyle = 'red';
            for (let i = 0; i < backgroundPoints.length; i++) {
                let x = backgroundPoints[i].x;
                let y = backgroundPoints[i].y;
                
                sourceCtx.beginPath();
                sourceCtx.arc(x, y, 5, 0, 2 * Math.PI);
                sourceCtx.fill();
            }
        }
      },
  }
});

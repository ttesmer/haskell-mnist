var canvas, ctx, flag, dot_flag = false;
var prevX, prevY, currX, currY = 0.0;

function init() {
    canvas = document.querySelector('#can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.clearRect(0.0, 0.0, w, h);
}

function findxy(res, e) {
    viewport = canvas.getBoundingClientRect();
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = (e.clientX - viewport.x) / 16.0;
        currY = (e.clientY - viewport.y) / 16.0;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = 'black';
            ctx.fillRect(currX, currY, 1, 1);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = (e.clientX - viewport.x) / 16.0;
            currY = (e.clientY - viewport.y) / 16.0;
            draw();
        }
    }
}

function forward() {
    imageData = ctx.getImageData(0, 0, 28, 28);
    return imageData.data; // returns Uint8ClampedArray
    /*ctx2 = document.getElementById("can2").getContext("2d");
    ctx2.putImageData(imageData, 0, 0);
    console.log(imageData.data); */ 
}
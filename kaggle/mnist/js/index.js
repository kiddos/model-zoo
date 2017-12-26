$(document).ready(function() {
  'use strict';

  const canvas = $('#canvas').get(0);
  const context = canvas.getContext('2d');
  const boundingRect = canvas.getBoundingClientRect();
  var pressed = false;
  var startX, startY;

  function clear() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
  }

  function draw(x1, y1, x2, y2, width = 6) {
    context.fillStyle = '#FFF';
    context.strokeStyle = '#FFF';

    context.lineWidth = 1;
    context.beginPath();
    context.arc(x1, y1, width, 0, Math.PI * 2, false);
    context.fill();
    context.closePath();

    context.beginPath();
    context.lineWidth = width * 2;
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    context.closePath();

    context.lineWidth = 1;
    context.beginPath();
    context.arc(x2, y2, width, 0, Math.PI * 2, false);
    context.fill();
    context.closePath();
  }

  $('#canvas').on('mousedown', function(event) {
    startX = event.pageX - boundingRect.left;
    startY = event.pageY - boundingRect.top;
    pressed = true;
  });

  $('#canvas').on('mouseup', function(event) {
    pressed = false;
  });

  $('#canvas').on('mousemove', function(event) {
    if (pressed) {
      var x = event.pageX - boundingRect.left;
      var y = event.pageY - boundingRect.top;
      draw(startX, startY, x, y);
      startX = x;
      startY = y;
    }
  });

  $('#canvas').on('mouseleave', function() {
    pressed = false;
  });

  clear();

  $('#btn-clear').on('click', function() {
    clear();
  });

  function sendImage() {
    var data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    $.post('/image', {image: data.toString()}, function(status) {
      console.log(status);
    });
  }

  sendImage();
});

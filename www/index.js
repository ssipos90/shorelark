import * as sim from "lib-simulation-wasm";

const simulation = new sim.Simulation();
const world = simulation.world();
console.log(world);

const viewport = document.getElementById('viewport');

const viewportWidth = viewport.width;
const viewportHeight = viewport.height;

const viewportScale = window.devicePixelRatio || 1;

const ctxt = viewport.getContext('2d');

// Automatically scales all operations by `viewportScale` - otherwise
// we'd have to `* viewportScale` everything by hand
// ctxt.scale(viewportScale, viewportScale);

// Rest of the code follows without any changes
ctxt.fillStyle = 'rgb(0, 0, 0)';

function drawTriangle(ctxt, x, y, size, rotation) {
  ctxt.beginPath();
  const initial = [
    x + Math.cos(rotation) * size * 1.5,
    y + Math.sin(rotation) * size * 1.5,
  ];
  ctxt.moveTo(
    ...initial
  );
  ctxt.lineTo(
    x + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
    y + Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
  );

  ctxt.lineTo(
    x + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
    y + Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
  );

  ctxt.lineTo(...initial);

  ctxt.stroke();
  ctxt.fillStyle = 'rgb(255, 255, 255)'; // A nice white color
  ctxt.fill();
}

function drawCircle(ctxt, x, y, radius) {
  ctxt.beginPath();

  ctxt.arc(x, y, radius, 0, 2.0 * Math.PI);

  ctxt.fillStyle = 'rgb(0, 255, 128)';
  ctxt.fill();
};

function redraw() {
  ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

  simulation.step();

  const world = simulation.world();
  for (const food of world.foods) {
    drawCircle(
      ctxt,
      food.x * viewportWidth,
      food.y * viewportHeight,
      (0.01 / 2.0) * viewportWidth,
    );
  }
  for (const animal of world.animals) {
    drawTriangle(
      ctxt,
      animal.x * viewportWidth,
      animal.y * viewportHeight,
      0.01 * viewportWidth,
      animal.rotation,
    );
  }

  requestAnimationFrame(redraw);
}

redraw();

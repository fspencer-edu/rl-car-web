const WORLD_W = 10;
const WORLD_H = 10;
const GRID = 20;
const MAX_STEPS = 120;
const SPEED = 0.45;

const START = [1, 1];
const GOAL = [8.5, 8.5];
const GOAL_RADIUS = 0.7;

const ACTIONS = [
  [0, 1],
  [0.7, 0.7],
  [1, 0],
  [0.7, -0.7],
  [0, -1],
  [-0.7, -0.7],
  [-1, 0],
  [-0.7, 0.7],
];

const OBSTACLES = [
  [2.0, 2.5, 1.2, 4.0],
  [4.5, 0.5, 1.0, 5.0],
  [6.3, 4.5, 1.3, 4.0],
  [2.5, 7.0, 3.0, 1.0],
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function norm(v) {
  const d = Math.hypot(v[0], v[1]) || 1;
  return [v[0] / d, v[1] / d];
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function inBounds(p) {
  return p[0] >= 0 && p[0] <= WORLD_W && p[1] >= 0 && p[1] <= WORLD_H;
}

function hitsObstacle(p) {
  return OBSTACLES.some(([x, y, w, h]) =>
    p[0] >= x && p[0] <= x + w && p[1] >= y && p[1] <= y + h
  );
}

function reachedGoal(p) {
  return distance(p, GOAL) <= GOAL_RADIUS;
}

function toState(pos) {
  const gx = clamp(Math.floor((pos[0] / WORLD_W) * GRID), 0, GRID - 1);
  const gy = clamp(Math.floor((pos[1] / WORLD_H) * GRID), 0, GRID - 1);
  return gy * GRID + gx;
}

function makeQTable() {
  return Array.from({ length: GRID * GRID }, () => Array(ACTIONS.length).fill(0));
}

function stepWorld(pos, actionIdx) {
  const oldDist = distance(pos, GOAL);
  const move = norm(ACTIONS[actionIdx]);
  const candidate = [pos[0] + move[0] * SPEED, pos[1] + move[1] * SPEED];

  let reward = -0.15;
  let nextPos = pos.slice();
  let done = false;

  if (!inBounds(candidate)) {
    reward = -4;
  } else if (hitsObstacle(candidate)) {
    reward = -6;
  } else {
    const newDist = distance(candidate, GOAL);
    reward += (oldDist - newDist) * 1.8;
    nextPos = candidate;
  }

  if (reachedGoal(nextPos)) {
    reward = 25;
    done = true;
  }

  return { nextPos, reward, done, heading: Math.atan2(move[1], move[0]) };
}

function movingAverage(arr, window = 20) {
  if (!arr.length) return [];
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = arr.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

let qTable = makeQTable();
let episode = 0;
let step = 0;
let epsilon = 1.0;
let pos = START.slice();
let path = [START.slice()];
let heading = 0;
let episodeReward = 0;
let rewards = [];
let stepsHistory = [];
let successHistory = [];
let running = false;
let animId = null;

const worldCanvas = document.getElementById("worldCanvas");
const rewardCanvas = document.getElementById("rewardCanvas");
const stepsCanvas = document.getElementById("stepsCanvas");
const successCanvas = document.getElementById("successCanvas");

const worldCtx = worldCanvas.getContext("2d");
const rewardCtx = rewardCanvas.getContext("2d");
const stepsCtx = stepsCanvas.getContext("2d");
const successCtx = successCanvas.getContext("2d");

function updateStats() {
  document.getElementById("episode").textContent = episode;
  document.getElementById("step").textContent = step;
  document.getElementById("epsilon").textContent = epsilon.toFixed(2);

  const recent = successHistory.slice(-25);
  const rate = recent.length ? (recent.reduce((a, b) => a + b, 0) / recent.length) * 100 : 0;
  document.getElementById("successRate").textContent = `${rate.toFixed(0)}%`;
}

function pickAction(state) {
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * ACTIONS.length);
  }
  const row = qTable[state];
  let best = 0;
  for (let i = 1; i < row.length; i++) {
    if (row[i] > row[best]) best = i;
  }
  return best;
}

function bestValue(state) {
  return Math.max(...qTable[state]);
}

function trainStep() {
  const alpha = 0.18;
  const gamma = 0.96;

  const state = toState(pos);
  const action = pickAction(state);
  const result = stepWorld(pos, action);
  const nextState = toState(result.nextPos);

  const tdTarget = result.reward + gamma * bestValue(nextState) * (result.done ? 0 : 1);
  qTable[state][action] += alpha * (tdTarget - qTable[state][action]);

  pos = result.nextPos;
  heading = result.heading;
  path.push(pos.slice());
  step += 1;
  episodeReward += result.reward;

  if (result.done || step >= MAX_STEPS) {
    rewards.push(episodeReward);
    stepsHistory.push(step);
    successHistory.push(result.done ? 1 : 0);

    episode += 1;
    epsilon = Math.max(0.05, epsilon * 0.992);
    step = 0;
    pos = START.slice();
    path = [START.slice()];
    heading = 0;
    episodeReward = 0;
  }
}

function drawWorld() {
  const w = worldCanvas.width;
  const h = worldCanvas.height;

  worldCtx.clearRect(0, 0, w, h);

  const sx = (x) => (x / WORLD_W) * w;
  const sy = (y) => h - (y / WORLD_H) * h;

  worldCtx.strokeStyle = "#e5e7eb";
  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * w;
    const y = (i / 10) * h;
    worldCtx.beginPath();
    worldCtx.moveTo(x, 0);
    worldCtx.lineTo(x, h);
    worldCtx.stroke();

    worldCtx.beginPath();
    worldCtx.moveTo(0, y);
    worldCtx.lineTo(w, y);
    worldCtx.stroke();
  }

  worldCtx.fillStyle = "rgba(0,0,0,0.2)";
  for (const [x, y, ow, oh] of OBSTACLES) {
    worldCtx.fillRect(sx(x), sy(y + oh), (ow / WORLD_W) * w, (oh / WORLD_H) * h);
  }

  worldCtx.beginPath();
  worldCtx.fillStyle = "rgba(34,197,94,0.25)";
  worldCtx.arc(sx(GOAL[0]), sy(GOAL[1]), (GOAL_RADIUS / WORLD_W) * w, 0, Math.PI * 2);
  worldCtx.fill();

  worldCtx.fillStyle = "#111827";
  worldCtx.font = "12px Arial";
  worldCtx.fillText("GOAL", sx(GOAL[0]) - 16, sy(GOAL[1]) + 4);

  worldCtx.beginPath();
  worldCtx.fillStyle = "rgba(59,130,246,0.3)";
  worldCtx.arc(sx(START[0]), sy(START[1]), 7, 0, Math.PI * 2);
  worldCtx.fill();

  if (path.length > 1) {
    worldCtx.beginPath();
    worldCtx.strokeStyle = "#2563eb";
    worldCtx.lineWidth = 2;
    worldCtx.moveTo(sx(path[0][0]), sy(path[0][1]));
    for (const p of path) {
      worldCtx.lineTo(sx(p[0]), sy(p[1]));
    }
    worldCtx.stroke();
  }

  const cx = sx(pos[0]);
  const cy = sy(pos[1]);

  worldCtx.save();
  worldCtx.translate(cx, cy);
  worldCtx.rotate(-heading);

  worldCtx.fillStyle = "#111827";
  worldCtx.fillRect(-12, -7, 24, 14);

  worldCtx.beginPath();
  worldCtx.moveTo(12, 0);
  worldCtx.lineTo(22, -5);
  worldCtx.lineTo(22, 5);
  worldCtx.closePath();
  worldCtx.fill();

  worldCtx.restore();
}

function drawLineChart(ctx, canvas, data, color) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!data.length) return;

  const pad = 20;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;

  ctx.strokeStyle = "#d1d5db";
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((v, i) => {
    const x = pad + (i / Math.max(1, data.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function redraw() {
  updateStats();
  drawWorld();
  drawLineChart(rewardCtx, rewardCanvas, movingAverage(rewards), "#2563eb");
  drawLineChart(stepsCtx, stepsCanvas, movingAverage(stepsHistory), "#7c3aed");
  drawLineChart(successCtx, successCanvas, movingAverage(successHistory).map(v => v * 100), "#16a34a");
}

function loop() {
  if (!running) return;
  trainStep();
  redraw();
  animId = requestAnimationFrame(loop);
}

document.getElementById("startBtn").addEventListener("click", () => {
  if (!running) {
    running = true;
    loop();
  }
});

document.getElementById("pauseBtn").addEventListener("click", () => {
  running = false;
  if (animId) cancelAnimationFrame(animId);
});

document.getElementById("resetBtn").addEventListener("click", () => {
  running = false;
  if (animId) cancelAnimationFrame(animId);

  qTable = makeQTable();
  episode = 0;
  step = 0;
  epsilon = 1.0;
  pos = START.slice();
  path = [START.slice()];
  heading = 0;
  episodeReward = 0;
  rewards = [];
  stepsHistory = [];
  successHistory = [];
  redraw();
});

redraw();
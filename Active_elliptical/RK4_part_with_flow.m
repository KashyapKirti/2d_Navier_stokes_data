clc; clear; close all;

% PARAMETERS
U0 = 1;
L = pi;
boxL = 2*L;
vs = 0.016;

alpha = 1.0; % alpha^2-1 / alpha^2+1
omch = 0.0;  % chirality
N = 10000;
tmax = 500; dt = 0.001;
tspan = 0:dt:tmax;

% INITIAL CONDITIONS
x = boxL * rand(N, 1);
y = boxL * rand(N, 1);
theta = 2*pi * rand(N, 1);
Dr = 0;
sigma = sqrt(2*Dr);
mu = 0;
x0 = 0;
% VIDEO SETUP
%v = VideoWriter('with_alignment.avi');
%v.FrameRate = 100;
%open(v);

% PLOT SETUP (fixed figure size 560x420 pixels)
% fig = figure('Position',[100 100 560 420]);
% axis([0 boxL 0 boxL]);
% axis square;
% hold on;
% ---- Functions for derivatives ----
% Flow + orientation dynamics
dynamics = @(x,y,theta) deal( ...
    U0 * sin(x - x0) .* cos(y) + vs * cos(theta), ...         % dx/dt
   -U0 * cos(x - x0) .* sin(y) + vs * sin(theta), ...         % dy/dt
    0.5 * alpha .* ((-U0 * sin(x - x0) .* sin(y)) - (-U0 * sin(x - x0) .* sin(y))) .* cos(2*theta) ...
  - alpha .* (U0 * cos(x - x0) .* cos(y)) .* sin(2*theta) ...
  - 0.5 * ((-U0 * sin(x - x0) .* sin(y)) + (-U0 * sin(x - x0) .* sin(y))) + omch ... % dtheta/dt
);

% ---- Main loop ----
for ti = 1:length(tspan)
    t = tspan(ti);

    % RK4 integration
    [k1x,k1y,k1th] = dynamics(x, y, theta);
    [k2x,k2y,k2th] = dynamics(x + 0.5*dt*k1x, y + 0.5*dt*k1y, theta + 0.5*dt*k1th);
    [k3x,k3y,k3th] = dynamics(x + 0.5*dt*k2x, y + 0.5*dt*k2y, theta + 0.5*dt*k2th);
    [k4x,k4y,k4th] = dynamics(x + dt*k3x, y + dt*k3y, theta + dt*k3th);

    % Update states
    x     = x     + dt/6 * (k1x + 2*k2x + 2*k3x + k4x);
    y     = y     + dt/6 * (k1y + 2*k2y + 2*k3y + k4y);
    theta = theta + dt/6 * (k1th + 2*k2th + 2*k3th + k4th) ...
                 + sqrt(dt) * sigma * randn(N,1); % noise in theta

    % Periodic boundary conditions
    x = mod(x, boxL);
    y = mod(y, boxL);

    if mod(ti,1000)==0
        fprintf("Step %d / %d\n",ti,length(tspan));
    end
end

% ---- Final scatter plot ----
scatter(x, y, 10, 'filled');
xlim([0 boxL]); ylim([0 boxL]);
xlabel('x'); ylabel('y');
axis square;

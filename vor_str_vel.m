clear; clc; close all;
n=256;
h=(2*pi)/n;
x_grid=(h:h:2*pi);
y_grid=(h:h:2*pi);
[X,Y]=meshgrid(x_grid,y_grid);

%%reading binary data

fid1=fopen('vor2500.dat');
dum1 = fread(fid1,1,'float32');
omega=fread(fid1,n*n, 'float64');
dum2 = fread(fid1,n,'float32');
omega = reshape(omega,[n,n]); 
omega=omega';

% Fourier wave numbers
kx = (2*pi/(n*h)) * [0:(n/2-1) -n/2:-1];
%kx  = fftshift((2*pi/(n*h)) * [-n/2 : n/2-1] ); 


ky = kx;
[KX,KY] = meshgrid(kx,ky);

k2 = KX.^2 + KY.^2;
k2(1,1) = 1;  % avoid division by zero at (0,0)

% FFT of vorticity
omega_hat = fft2(omega);

% Solve Poisson in Fourier space: psi_hat = -omega_hat / k^2
psi_hat = -omega_hat ./ k2;

% Force zero-mean (streamfunction defined up to a constant)
psi_hat(1,1) = 0;

% Inverse FFT to get streamfunction
psi = real(ifft2(psi_hat));

% psi = streamfunction field (n x n)
% h   = grid spacing
% x_grid, y_grid are your mesh arrays

% Compute derivatives
[dpsi_dx, dpsi_dy] = gradient(psi, h, h);
% Velocity components
u = dpsi_dy;      % u = dψ/dy
v = -dpsi_dx;     % v = -dψ/dx

%%
% ---------------- PLOTTING ----------------
figure;

% 1. Vorticity field
subplot(1,3,1);
imagesc(x_grid,y_grid,omega);
axis equal tight; colorbar;
title('\omega (vorticity)');
xlabel('x'); ylabel('y');
set(gca,'YDir','normal');

% 2. Streamfunction with streamlines
subplot(1,3,2);
imagesc(x_grid,y_grid,psi);
hold on;

h= streamslice(X,Y,u,v);   % make streamlines white
set(h,'Color','w','LineWidth',1.0);
hold off;

% hold off;
axis equal tight; colorbar;
title('\psi (streamfunction) with streamlines');
xlabel('x'); ylabel('y');
set(gca,'YDir','normal');

% 3. Vorticity with velocity quiver
subplot(1,3,3);
imagesc(x_grid,y_grid,sqrt(u.^2+v.^2)); 
hold on;
step = 10; % reduce arrow density
quiver(X(1:step:end,1:step:end), Y(1:step:end,1:step:end), ...
       u(1:step:end,1:step:end), v(1:step:end,1:step:end),1.5, 'k','LineWidth',1.0);
hold off;
%xlim([h,2*pi]); ylim([h,2*pi])
axis square; colorbar;
title('|v| with velocity vectors');
xlabel('x'); ylabel('y');
set(gca,'YDir','normal');
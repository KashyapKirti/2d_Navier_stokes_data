clear; clc;

N = 256;
L = 2*pi;
dx = L/N;

x = 0:dx:(L-dx);
y = 0:dx:(L-dx);
[X, Y] = meshgrid(x,y);

% --- wave numbers (Fourier modes) ---
kx = [0:(N/2-1)  -N/2:-1] * (2*pi/L);
ky = kx;
[KX, KY] = meshgrid(kx, ky);

% === Initialize particles ===
Np = 200;                      
xp = L * rand(Np,1);          
yp = L * rand(Np,1);          

dt = 0.05;    
nsteps = 10;  

% --- Video setup ---
v = VideoWriter('../../lagrangian_tracking.avi'); 
v.FrameRate = 5;   
open(v);

% --- loop over 20 snapshots ---
for n = 1:21
    it = 2500 + (n-1);   
    fname = ['str' num2str(it,'%d.dat')];   
    
    fid = fopen(fname,'r');
    if fid < 0
        error('File %s not found', fname);
    end
    dum = fread(fid,1,'float32');
    psi = fread(fid,[N N],'float64');
    dum = fread(fid,1,'float32');
    fclose(fid);
    
    psi = transpose(psi);   
    
    % --- FFT of streamfunction ---
    psi_hat = fft2(psi);
    
    % --- Velocity in Fourier space ---
    ux_hat = 1i * KY .* psi_hat;
    uy_hat = -1i * KX .* psi_hat;

    % --- Inverse FFT to real space ---
    ux = real(ifft2(ux_hat));
    uy = real(ifft2(uy_hat));

    % === Evolve particles for nsteps with this snapshot ===
    for t = 1:nsteps
        % interpolate velocities at particle positions
        vxp = interp2(X, Y, ux, xp, yp, 'linear');
        vyp = interp2(X, Y, uy, xp, yp, 'linear');
        
        % update positions
        xp = xp + vxp*dt;
        yp = yp + vyp*dt;
        
        % periodic BC
        xp = mod(xp, L);
        yp = mod(yp, L);
    end
    
    % --- plot --- %
    figure(1); clf;
    umag = sqrt(ux.^2 + uy.^2);
    imagesc(x,y,umag); 
    axis equal tight;
    set(gca,'YDir','normal');
    colormap('jet');
    colorbar;
    clim([0 0.5]);   

    hold on;
    % black streamlines
    h = streamslice(X,Y,ux,uy);
    set(h,'Color','k');   
    
    % particle positions
    plot(xp, yp, 'ro','MarkerFaceColor','r','MarkerSize',4);
    title(sprintf('Particles after snapshot %d', it));

    % capture frame for video
    frame = getframe(gcf);
    writeVideo(v, frame);
end

close(v);
disp('Video saved as particle_advection.avi');

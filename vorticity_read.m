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

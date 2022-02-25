clc, clear;
cd0 = matlab.desktop.editor.getActiveFilename;
dash = cd0(strfind(cd0,'SIMULATION_BEAD.m')-1);
cd0 = cd0(1:strfind(cd0,'SIMULATION_BEAD.m')-2);
addpath(genpath(cd0));
used_gpu_device=[];
gpu_device=gpuDevice(used_gpu_device);
%% set the simulation parameters
MULTI_GPU=false; % Use Multiple GPU?

%1 optical parameters
cubes_file="E:\Matlock\Simulation\tIDT\191102_PaperData\PhaseEval_Vkp\illum8\illumAngle_8pos_cube9_n=1.01_6ux6ux10p_n0=1_bkn=1_n(r)=1.01_f=0um.mat";
cubes_data=load(cubes_file).rcn;
cubes_data.n_s = padarray(cubes_data.n_s, [64 64], "replicate", "both");
params=BASIC_OPTICAL_PARAMETER();
params.NA=0.65; % Numerical aperture
params.RI_bg=1.; % Background RI
params.wavelength=cubes_data.lambda; % [um]
params.resolution=cubes_data.delta; % 3D Voxel size [um]
params.use_abbe_sine=true; % Abbe sine condition according to demagnification condition
params.vector_simulation=true;false; % True/false: dyadic/scalar Green's function
params.size=size(cubes_data.n_s); % 3D volume grid
%2 illumination parameters
field_generator_params=FIELD_GENERATOR.get_default_parameters(params);
field_generator_params.illumination_number=8; 
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
field_generator_params.start_with_normal=false;
field_generator_params.percentage_NA_usage=0.99;

%3 phantom generation parameter
phantom_params=PHANTOM.get_default_parameters();
phantom_params.name='bead';%'RBC';
RI_sp=1.01;
phantom_params.outer_size=params.size;
phantom_params.inner_size=round(ones(1,3) * 5 ./ params.resolution);
phantom_params.rotation_angles = [0 0 0];


%4 forward solver parameters
forward_params=FORWARD_SOLVER_CONVERGENT_BORN.get_default_parameters(params);
forward_params.use_GPU=false;
%5 multiple scattering solver
if ~MULTI_GPU
    backward_params=BACKWARD_SOLVER_MULTI.get_default_parameters(params);
else
    backward_params=BACKWARD_SOLVER_MULTI_MULTI_GPU.get_default_parameters(params);
end


forward_params_backward=FORWARD_SOLVER_CONVERGENT_BORN.get_default_parameters(forward_params);
forward_params_backward.return_transmission=true;
forward_params_backward.return_reflection=true;
forward_params_backward.return_3D=true;
forward_params_backward.boundary_thickness=[2 2 4]; % if xy is nonzero, acyclic convolution is applied.
forward_params_backward.used_gpu = 0;
%6 parameter for rytov solver


%% create phantom and solve the forward problem
% make the phantom
RI=cubes_data.n_s;
%create the incident field
field_generator=FIELD_GENERATOR(field_generator_params);
input_field=field_generator.get_fields();
%compute the forward field - CBS
forward_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_params);
forward_solver.set_RI(RI);
tic;
[field_trans,field_ref,field_3D]=forward_solver.solve(input_field);
toc;

% Display results: transmitted field
[input_field_scalar,field_trans_scalar]=vector2scalarfield(input_field,field_trans);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));
figure;orthosliceViewer(squeeze(abs(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap gray; title('Amplitude')
figure;orthosliceViewer(squeeze(angle(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap jet; title('Phase')

%plot camera trajectories
figure;
hold on
%plot_trajectory(file,color)
poses = readtable('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_desk/groundtruth.txt','Delimiter',' ','ReadVariableNames',true,'HeaderLines',2);
%poses = readtable('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_desk/desk_evaluation/poses_complete_pcl.txt','Delimiter',' ','ReadVariableNames',true,'HeaderLines',1);
for i = 1:size(poses,1)
    if (mod(i,25) == 0)
    %currPose = poses{1,2:8};
    t = poses{i,2:4};
    %reorder the quaternion
    %q = ;
    R = quat2rotm(poses{i,[8,5,6,7]});
    %currPose = [quat2rotm(q_start),t_start';0,0,0,1];
    plotCamera('Location',t,'Orientation',R,'Opacity',0,'Size',0.01,'Label',num2str(i));
    end
end
hold off
    
# import open3d as o3d
# import numpy as np
# import copy
# from sklearn.cluster import DBSCAN


# ##################################################
# # PATHS
# ##################################################

# geom_path = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010_nom.STL"

# nominal_icp = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010_nom.STL"
# damaged_icp = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010001.STL"


# ##################################################
# # SCREENSHOT FUNCTION
# ##################################################

# def capture(geometries, filename,
#             width=1600,
#             height=1200):

#     vis=o3d.visualization.Visualizer()

#     vis.create_window(
#         visible=True,
#         width=width,
#         height=height
#     )

#     for g in geometries:
#         vis.add_geometry(g)

#     opt=vis.get_render_option()

#     opt.background_color=np.asarray([1,1,1])

#     opt.point_size=6

#     opt.mesh_show_wireframe=True
#     opt.mesh_show_back_face=True

#     # optional, usually looks better
#     opt.light_on=True

#     ctr=vis.get_view_control()

#     ctr.set_front([1,-1,1])

#     ctr.set_up([0,0,1])

#     # ctr.set_front([0,0.95,-0.5])
#     # ctr.set_up([0,1,0])

#     ctr.set_zoom(.72)

#     vis.poll_events()
#     vis.update_renderer()

#     vis.capture_screen_image(filename)

#     vis.destroy_window()

#     print("saved:", filename)


# ##################################################
# # 1 GEOMETRY REPRESENTATIONS
# ##################################################

# def geometry_types():

#     mesh=o3d.io.read_triangle_mesh(geom_path)

#     mesh.compute_vertex_normals()


#     ##################################
#     # POINT CLOUD
#     ##################################

#     # lower density
#     pcd=mesh.sample_points_uniformly(
#         number_of_points=8000
#     )

#     # grey
#     pcd.paint_uniform_color(
#         [0.6,0.6,0.6]
#     )

#     capture(
#         [pcd],
#         "fig_pointcloud.png"
#     )



#     ##################################
#     # MESH
#     ##################################

#     mesh.paint_uniform_color(
#         [1.0,0.55,0.0]   # orange
#     )

#     capture(
#         [mesh],
#         "fig_mesh.png"
#     )



#     ##################################
#     # VOXELS (solid blue cubes)
#     ##################################

#     # Large voxels = fewer cubes
#     voxel_size = 7.5

#     voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
#         mesh,
#         voxel_size=voxel_size
#     )

#     # Convert each voxel to a solid cube mesh
#     voxels_as_mesh=[]

#     for v in voxel_grid.get_voxels():

#         cube=o3d.geometry.TriangleMesh.create_box(
#             width=voxel_size,
#             height=voxel_size,
#             depth=voxel_size
#         )

#         # move cube to voxel location
#         cube.translate(
#             voxel_grid.origin + (
#                 np.array(v.grid_index)*voxel_size
#             )
#         )

#         # blue cubes
#         cube.paint_uniform_color(
#             [0.25,0.50,1.0]
#         )

#         cube.compute_vertex_normals()

#         voxels_as_mesh.append(cube)


#     # Merge cubes into one mesh
#     voxel_mesh=voxels_as_mesh[0]

#     for c in voxels_as_mesh[1:]:
#         voxel_mesh += c


#     capture(
#         [voxel_mesh],
#         "fig_voxel.png"
#     )

# ##################################################
# # 2 ICP ALIGNMENT
# ##################################################

# def icp_figure():

#     nom=o3d.io.read_triangle_mesh(nominal_icp)
#     dmg=o3d.io.read_triangle_mesh(damaged_icp)

#     source=nom.sample_points_uniformly(8000)
#     target=dmg.sample_points_uniformly(8000)

#     source=source.voxel_down_sample(1.5)
#     target=target.voxel_down_sample(1.5)


#     # source.paint_uniform_color([1,0,0])

#     # target.paint_uniform_color([0,0,1])

#     # Source = blue
#     source.paint_uniform_color(
#         [0.6,0.6,0.6]
#     )

#     # Target = green
#     target.paint_uniform_color(
#         [0.1,0.5,1.0]
#     )


#     # misalign manually

#     # T=np.eye(4)

#     # T[:3,3]=[30,15,0]

#     # source.transform(T)

#     # --------------------------------
#     # Manual misalignment:
#     # rotation + translation
#     # --------------------------------

#     T=np.eye(4)

#     # Rotation (degrees)
#     rx=np.radians(12)
#     ry=np.radians(-8)
#     rz=np.radians(18)

#     # Rotation matrices
#     Rx=np.array([
#     [1,0,0],
#     [0,np.cos(rx),-np.sin(rx)],
#     [0,np.sin(rx), np.cos(rx)]
#     ])

#     Ry=np.array([
#     [np.cos(ry),0,np.sin(ry)],
#     [0,1,0],
#     [-np.sin(ry),0,np.cos(ry)]
#     ])

#     Rz=np.array([
#     [np.cos(rz),-np.sin(rz),0],
#     [np.sin(rz), np.cos(rz),0],
#     [0,0,1]
#     ])

#     # Combined rotation
#     R = Rz @ Ry @ Rx

#     T[:3,:3]=R

#     # Translation
#     T[:3,3]=[50,15,0]

#     # Apply rigid transform
#     source.transform(T)


#     capture(
#         [source,target],
#         "fig_icp_before.png"
#     )


#     # ICP

#     reg=o3d.pipelines.registration.registration_icp(
#         source,
#         target,
#         25.0,
#         np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )

#     aligned=copy.deepcopy(source)

#     aligned.transform(
#         reg.transformation
#     )

#     capture(
#         [aligned,target],
#         "fig_icp_after.png"
#     )



# ##################################################
# # 3 BOOLEAN DIFFERENCE CONCEPT
# ##################################################

# def boolean_concept():

#     # ----------------------------------
#     # Load real components
#     # ----------------------------------

#     nom=o3d.io.read_triangle_mesh(
#         nominal_icp
#     )

#     dmg=o3d.io.read_triangle_mesh(
#         damaged_icp
#     )

#     # Sample clouds
#     nom_pcd=nom.sample_points_uniformly(
#         8000
#     )

#     dmg_pcd=dmg.sample_points_uniformly(
#         8000
#     )


#     # ----------------------------------
#     # ICP ALIGN first
#     # (important for meaningful boolean)
#     # ----------------------------------

#     reg=o3d.pipelines.registration.registration_icp(
#         dmg_pcd,
#         nom_pcd,
#         25.0,
#         np.eye(4),
#         o3d.pipelines.registration.
#         TransformationEstimationPointToPoint()
#     )

#     dmg_pcd.transform(
#         reg.transformation
#     )


#     # ----------------------------------
#     # Boolean difference approximation
#     # Missing points in nominal that do
#     # not exist in damaged
#     # ----------------------------------

#     nom_pts=np.asarray(
#         nom_pcd.points
#     )

#     dmg_pts=np.asarray(
#         dmg_pcd.points
#     )

#     # nearest neighbor
#     from scipy.spatial import cKDTree

#     tree=cKDTree(
#         dmg_pts
#     )

#     dists,_=tree.query(
#         nom_pts,
#         k=1
#     )

#     # threshold controls "missing"
#     missing_idx=np.where(
#         dists>10.5
#     )[0]

#     missing=o3d.geometry.PointCloud()

#     missing.points=o3d.utility.Vector3dVector(
#         nom_pts[missing_idx]
#     )


#     # ----------------------------------
#     # Colors
#     # ----------------------------------

#     nom_pcd.paint_uniform_color(
#         [0.6,0.6,0.6]
#     )

#     dmg_pcd.paint_uniform_color(
#         [0.1,0.5,1.0]
#     )

#     missing.paint_uniform_color(
#         [1,0,0]
#     )


#     # ----------------------------------
#     # Capture figures
#     # ----------------------------------

#     # Nominal
#     capture(
#         [nom_pcd],
#         "fig_boolean_nominal.png"
#     )

#     # Damaged
#     capture(
#         [dmg_pcd],
#         "fig_boolean_damaged.png"
#     )

#     # Missing region
#     capture(
#         [dmg_pcd,missing],
#         "fig_boolean_missing.png"
#     )


# # ##################################################
# # # 4 DBSCAN
# # ##################################################

# # def dbscan_figure():

# #     np.random.seed(2)

# #     c1=np.random.randn(300,3)+[0,0,0]
# #     c2=np.random.randn(300,3)+[10,0,0]
# #     c3=np.random.randn(300,3)+[0,10,0]

# #     noise=np.random.uniform(
# #         -8,
# #         15,
# #         (80,3)
# #     )

# #     X=np.vstack([c1,c2,c3,noise])


# #     pcd=o3d.geometry.PointCloud()

# #     pcd.points=o3d.utility.Vector3dVector(X)

# #     pcd.paint_uniform_color([0,0,0])

# #     capture(
# #         [pcd],
# #         "fig_dbscan_before.png"
# #     )


# #     labels=np.array(
# #         pcd.cluster_dbscan(
# #             eps=1.5,
# #             min_points=10
# #         )
# #     )

# #     colors=np.zeros(
# #         (len(labels),3)
# #     )

# #     cmap=np.array([
# #         [1,0,0],
# #         [0,1,0],
# #         [0,0,1],
# #         [1,0,1]
# #     ])

# #     for i in np.unique(labels):

# #         if i==-1:

# #             colors[labels==-1]=[.6,.6,.6]

# #         else:

# #             colors[labels==i]=cmap[
# #                 i%4
# #             ]


# #     pcd.colors=o3d.utility.Vector3dVector(
# #         colors
# #     )

# #     capture(
# #         [pcd],
# #         "fig_dbscan_after.png"
# #     )

# ##################################################
# # 4 DBSCAN
# ##################################################

# def dbscan_figure():

#     import matplotlib.pyplot as plt
#     from sklearn.cluster import DBSCAN

#     np.random.seed(4)

#     ##################################
#     # Simulated sparse cloud
#     ##################################

#     c1=np.random.randn(150,3)*0.7+[0,0,0]
#     c2=np.random.randn(150,3)*0.7+[5,0,0]
#     c3=np.random.randn(150,3)*0.7+[2.5,4,0]
#     c4=np.random.randn(150,3)*0.7+[1,2,3]

#     # NEW extra clusters
#     c5=np.random.randn(150,3)*0.7+[6,4,2]
#     c6=np.random.randn(150,3)*0.7+[3,-2,4]

#     noise=np.random.uniform(
#         -3,
#         8,
#         (60,3)
#     )

#     X=np.vstack(
#         [
#         c1,c2,c3,c4,
#         c5,c6,
#         noise
#         ]
#     )


#     ##################################
#     # FIGURE 1
#     ##################################

#     fig=plt.figure(
#         figsize=(8,7)
#     )

#     ax=fig.add_subplot(
#         projection='3d'
#     )

#     ax.scatter(
#         X[:,0],
#         X[:,1],
#         X[:,2],
#         s=12,
#         c='dimgray'
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     # More isometric / seen from above
#     ax.view_init(
#         elev=35,
#         azim=45
#     )

#     plt.tight_layout()

#     plt.savefig(
#         "fig_dbscan_before.png",
#         dpi=300,
#         bbox_inches='tight'
#     )

#     plt.close()



#     ##################################
#     # DBSCAN
#     ##################################

#     labels=DBSCAN(
#         eps=0.75,
#         min_samples=10
#     ).fit_predict(X)



#     ##################################
#     # FIGURE 2
#     ##################################

#     cluster_colors=[
#         'red',
#         'blue',
#         'green',
#         'orange',
#         'purple',
#         'cyan'
#     ]


#     fig=plt.figure(
#         figsize=(8,7)
#     )

#     ax=fig.add_subplot(
#         projection='3d'
#     )


#     for i in np.unique(labels):

#         mask=labels==i

#         if i==-1:

#             ax.scatter(
#                 X[mask,0],
#                 X[mask,1],
#                 X[mask,2],
#                 c='gray',
#                 s=12
#             )

#         else:

#             ax.scatter(
#                 X[mask,0],
#                 X[mask,1],
#                 X[mask,2],
#                 c=cluster_colors[
#                     i % len(cluster_colors)
#                 ],
#                 s=18
#             )


#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     ax.view_init(
#         elev=35,
#         azim=45
#     )

#     plt.tight_layout()

#     plt.savefig(
#         "fig_dbscan_after.png",
#         dpi=300,
#         bbox_inches='tight'
#     )

#     plt.close()

#     print("Saved DBSCAN figures")


# ##################################################
# # 5 RANSAC CONCEPTS
# ##################################################

# def ransac_figures():

#     import matplotlib.pyplot as plt

#     np.random.seed(7)

#     elev=35
#     azim=45


#     ########################################
#     # 1 PLANE
#     ########################################

#     x=np.random.uniform(-4,4,500)
#     y=np.random.uniform(-4,4,500)

#     z=0.5*x + 0.2*y + np.random.normal(
#         0,
#         0.3,
#         500
#     )

#     noise=np.random.uniform(
#         -5,
#         5,
#         (80,3)
#     )

#     pts=np.column_stack(
#         [x,y,z]
#     )

#     cloud=np.vstack(
#         [pts,noise]
#     )


#     # BEFORE
#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_plane_before.png",
#         dpi=300
#     )
#     plt.close()


#     # AFTER (fit plane)
#     xx,yy=np.meshgrid(
#         np.linspace(-4,4,20),
#         np.linspace(-4,4,20)
#     )

#     zz=0.5*xx + 0.2*yy

#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.plot_surface(
#         xx,yy,zz,
#         alpha=0.5,
#         color='red'
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_plane_after.png",
#         dpi=300
#     )
#     plt.close()



#     ########################################
#     # 2 SPHERE
#     ########################################

#     phi=np.random.uniform(
#         0,np.pi,500
#     )

#     theta=np.random.uniform(
#         0,2*np.pi,500
#     )

#     r=3

#     x=r*np.sin(phi)*np.cos(theta)
#     y=r*np.sin(phi)*np.sin(theta)
#     z=r*np.cos(phi)

#     pts=np.column_stack(
#         [x,y,z]
#     )

#     cloud=np.vstack(
#         [pts,noise]
#     )


#     # BEFORE
#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_sphere_before.png",
#         dpi=300
#     )
#     plt.close()


#     # AFTER
#     u,v=np.mgrid[
#         0:2*np.pi:20j,
#         0:np.pi:20j
#     ]

#     xs=3*np.cos(u)*np.sin(v)
#     ys=3*np.sin(u)*np.sin(v)
#     zs=3*np.cos(v)

#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.plot_surface(
#         xs,ys,zs,
#         alpha=0.5,
#         color='blue'
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_sphere_after.png",
#         dpi=300
#     )
#     plt.close()



#     ########################################
#     # 3 CYLINDER
#     ########################################

#     th=np.random.uniform(
#         0,
#         2*np.pi,
#         500
#     )

#     z=np.random.uniform(
#         -4,
#         4,
#         500
#     )

#     x=2*np.cos(th)
#     y=2*np.sin(th)

#     pts=np.column_stack(
#         [x,y,z]
#     )

#     cloud=np.vstack(
#         [pts,noise]
#     )


#     # BEFORE
#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_cylinder_before.png",
#         dpi=300
#     )
#     plt.close()


#     # AFTER
#     th,z=np.meshgrid(
#         np.linspace(0,2*np.pi,30),
#         np.linspace(-4,4,30)
#     )

#     xc=2*np.cos(th)
#     yc=2*np.sin(th)

#     fig=plt.figure(figsize=(8,7))
#     ax=fig.add_subplot(projection='3d')

#     ax.scatter(
#         cloud[:,0],
#         cloud[:,1],
#         cloud[:,2],
#         c='dimgray',
#         s=10
#     )

#     ax.plot_surface(
#         xc,yc,z,
#         alpha=0.5,
#         color='green'
#     )

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=elev,azim=azim)

#     plt.tight_layout()
#     plt.savefig(
#         "fig_ransac_cylinder_after.png",
#         dpi=300
#     )
#     plt.close()


#     print("Saved RANSAC figures")



# ##################################################

# if __name__=="__main__":

#     geometry_types()

#     icp_figure()

#     boolean_concept()

#     dbscan_figure()

#     ransac_figures()

import open3d as o3d
import numpy as np
import copy
from sklearn.cluster import DBSCAN


##################################################
# PATHS
##################################################

geom_path = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010_nom.STL"

nominal_icp = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010_nom.STL"
damaged_icp = "/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/010/010001.STL"


##################################################
# SCREENSHOT FUNCTION
##################################################

def capture(geometries, filename,
            width=1600,
            height=1200):

    vis=o3d.visualization.Visualizer()

    vis.create_window(
        visible=True,
        width=width,
        height=height
    )

    for g in geometries:
        vis.add_geometry(g)

    opt=vis.get_render_option()

    opt.background_color=np.asarray([1,1,1])

    opt.point_size=6

    opt.mesh_show_wireframe=True
    opt.mesh_show_back_face=True

    # optional, usually looks better
    opt.light_on=True

    ctr=vis.get_view_control()

    ctr.set_front([1,-1,1])

    ctr.set_up([0,0,1])

    # ctr.set_front([0,0.95,-0.5])
    # ctr.set_up([0,1,0])

    ctr.set_zoom(.72)

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(filename)

    vis.destroy_window()

    print("saved:", filename)


##################################################
# 1 GEOMETRY REPRESENTATIONS
##################################################

def geometry_types():

    import matplotlib.pyplot as plt

    mesh=o3d.io.read_triangle_mesh(
        geom_path
    )

    verts=np.asarray(
        mesh.vertices
    )

    tris=np.asarray(
        mesh.triangles
    )


    ##################################
    # POINT CLOUD
    ##################################

    pcd=mesh.sample_points_uniformly(
        3000
    )

    pts=np.asarray(
        pcd.points
    )

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        c='dimgray',
        s=6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_pointcloud.png",
        dpi=300
    )

    plt.close()



    ##################################
    # MESH
    ##################################

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.plot_trisurf(
        verts[:,0],
        verts[:,1],
        verts[:,2],
        triangles=tris,
        color='orange',
        edgecolor='k',
        linewidth=.2
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        verts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_mesh.png",
        dpi=300
    )

    plt.close()



    ##################################
    # VOXELS
    ##################################

    from mpl_toolkits.mplot3d import Axes3D

    voxel_size=7.5

    voxel_grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size
    )

    vox=np.array([
        v.grid_index
        for v in voxel_grid.get_voxels()
    ])

    # Build occupancy grid for matplotlib voxels()
    ix=vox[:,0]-vox[:,0].min()
    iy=vox[:,1]-vox[:,1].min()
    iz=vox[:,2]-vox[:,2].min()

    grid=np.zeros(
        (
            ix.max()+1,
            iy.max()+1,
            iz.max()+1
        ),
        dtype=bool
    )

    grid[
        ix,
        iy,
        iz
    ]=True


    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    # TRUE voxel cubes
    ax.voxels(
        grid,
        facecolors='dodgerblue',
        edgecolor='black',     # <-- black cube edges
        linewidth=0.6          # <-- makes edges visible
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    # equal scaling for voxel grid indices
    set_axes_equal(
        ax,
        np.column_stack(
            [ix,iy,iz]
        )
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_voxel.png",
        dpi=300
    )

    plt.close()

##################################################
# 2 ICP ALIGNMENT
##################################################

def icp_figure():

    import matplotlib.pyplot as plt

    ##################################
    # Load models
    ##################################

    nom=o3d.io.read_triangle_mesh(
        nominal_icp
    )

    dmg=o3d.io.read_triangle_mesh(
        damaged_icp
    )

    source=nom.sample_points_uniformly(
        8000
    )

    target=dmg.sample_points_uniformly(
        8000
    )

    source=source.voxel_down_sample(1.5)
    target=target.voxel_down_sample(1.5)


    ##################################
    # Manual misalignment
    ##################################

    T=np.eye(4)

    rx=np.radians(12)
    ry=np.radians(-8)
    rz=np.radians(18)

    Rx=np.array([
    [1,0,0],
    [0,np.cos(rx),-np.sin(rx)],
    [0,np.sin(rx),np.cos(rx)]
    ])

    Ry=np.array([
    [np.cos(ry),0,np.sin(ry)],
    [0,1,0],
    [-np.sin(ry),0,np.cos(ry)]
    ])

    Rz=np.array([
    [np.cos(rz),-np.sin(rz),0],
    [np.sin(rz), np.cos(rz),0],
    [0,0,1]
    ])

    R=Rz@Ry@Rx

    T[:3,:3]=R

    T[:3,3]=[50,15,0]

    source.transform(T)


    ##################################
    # BEFORE ICP
    ##################################

    s=np.asarray(
        source.points
    )

    t=np.asarray(
        target.points
    )

    all_pts=np.vstack(
        [s,t]
    )

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        s[:,0],s[:,1],s[:,2],
        c='dimgray',
        s=6
    )

    ax.scatter(
        t[:,0],t[:,1],t[:,2],
        c='dodgerblue',
        s=6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        all_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_icp_before.png",
        dpi=300
    )

    plt.close()



    ##################################
    # ICP
    ##################################

    reg=o3d.pipelines.registration.registration_icp(
        source,
        target,
        25.0,
        np.eye(4),
        o3d.pipelines.registration.
        TransformationEstimationPointToPoint()
    )

    aligned=copy.deepcopy(
        source
    )

    aligned.transform(
        reg.transformation
    )


    ##################################
    # AFTER ICP
    ##################################

    a=np.asarray(
        aligned.points
    )

    all_pts=np.vstack(
        [a,t]
    )

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        a[:,0],a[:,1],a[:,2],
        c='dimgray',
        s=6
    )

    ax.scatter(
        t[:,0],t[:,1],t[:,2],
        c='dodgerblue',
        s=6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        all_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_icp_after.png",
        dpi=300
    )

    plt.close()

    print("Saved ICP figures")

##################################################
# 3 BOOLEAN DIFFERENCE CONCEPT
##################################################

def boolean_concept():

    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree


    ##################################
    # Load models
    ##################################

    nom=o3d.io.read_triangle_mesh(
        nominal_icp
    )

    dmg=o3d.io.read_triangle_mesh(
        damaged_icp
    )

    nom_pcd=nom.sample_points_uniformly(
        8000
    )

    dmg_pcd=dmg.sample_points_uniformly(
        8000
    )


    ##################################
    # ICP first
    ##################################

    reg=o3d.pipelines.registration.registration_icp(
        dmg_pcd,
        nom_pcd,
        25.0,
        np.eye(4),
        o3d.pipelines.registration.
        TransformationEstimationPointToPoint()
    )

    dmg_pcd.transform(
        reg.transformation
    )


    ##################################
    # Boolean difference
    ##################################

    nom_pts=np.asarray(
        nom_pcd.points
    )

    dmg_pts=np.asarray(
        dmg_pcd.points
    )

    tree=cKDTree(
        dmg_pts
    )

    dists,_=tree.query(
        nom_pts,
        k=1
    )

    # adjust if needed later
    missing_idx=np.where(
        dists > 10.5
    )[0]

    missing_pts=nom_pts[
        missing_idx
    ]


    ##################################
    # FIGURE 1
    # Nominal
    ##################################

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        nom_pts[:,0],
        nom_pts[:,1],
        nom_pts[:,2],
        c='dimgray',
        s=6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        nom_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_boolean_nominal.png",
        dpi=300
    )

    plt.close()



    ##################################
    # FIGURE 2
    # Damaged
    ##################################

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        dmg_pts[:,0],
        dmg_pts[:,1],
        dmg_pts[:,2],
        c='dodgerblue',
        s=6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        dmg_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_boolean_damaged.png",
        dpi=300
    )

    plt.close()



    ##################################
    # FIGURE 3
    # Missing volume
    ##################################

    all_pts=np.vstack(
        [dmg_pts,missing_pts]
    )

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        dmg_pts[:,0],
        dmg_pts[:,1],
        dmg_pts[:,2],
        c='lightgray',
        s=5
    )

    ax.scatter(
        missing_pts[:,0],
        missing_pts[:,1],
        missing_pts[:,2],
        c='red',
        s=8
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        all_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_boolean_missing.png",
        dpi=300
    )

    plt.close()

    print("Saved Boolean figures")
# ##################################################
# # 4 DBSCAN
# ##################################################

# def dbscan_figure():

#     np.random.seed(2)

#     c1=np.random.randn(300,3)+[0,0,0]
#     c2=np.random.randn(300,3)+[10,0,0]
#     c3=np.random.randn(300,3)+[0,10,0]

#     noise=np.random.uniform(
#         -8,
#         15,
#         (80,3)
#     )

#     X=np.vstack([c1,c2,c3,noise])


#     pcd=o3d.geometry.PointCloud()

#     pcd.points=o3d.utility.Vector3dVector(X)

#     pcd.paint_uniform_color([0,0,0])

#     capture(
#         [pcd],
#         "fig_dbscan_before.png"
#     )


#     labels=np.array(
#         pcd.cluster_dbscan(
#             eps=1.5,
#             min_points=10
#         )
#     )

#     colors=np.zeros(
#         (len(labels),3)
#     )

#     cmap=np.array([
#         [1,0,0],
#         [0,1,0],
#         [0,0,1],
#         [1,0,1]
#     ])

#     for i in np.unique(labels):

#         if i==-1:

#             colors[labels==-1]=[.6,.6,.6]

#         else:

#             colors[labels==i]=cmap[
#                 i%4
#             ]


#     pcd.colors=o3d.utility.Vector3dVector(
#         colors
#     )

#     capture(
#         [pcd],
#         "fig_dbscan_after.png"
#     )

##################################################
# 4 DBSCAN
##################################################

def dbscan_figure():

    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    np.random.seed(4)

    ##################################
    # Simulated sparse cloud
    ##################################

    c1=np.random.randn(150,3)*0.7+[0,0,0]
    c2=np.random.randn(150,3)*0.7+[5,0,0]
    c3=np.random.randn(150,3)*0.7+[2.5,4,0]
    c4=np.random.randn(150,3)*0.7+[1,2,3]

    # NEW extra clusters
    c5=np.random.randn(150,3)*0.7+[6,4,2]
    c6=np.random.randn(150,3)*0.7+[3,-2,4]

    noise=np.random.uniform(
        -3,
        8,
        (60,3)
    )

    X=np.vstack(
        [
        c1,c2,c3,c4,
        c5,c6,
        noise
        ]
    )


    ##################################
    # FIGURE 1
    ##################################

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )

    ax.scatter(
        X[:,0],
        X[:,1],
        X[:,2],
        s=12,
        c='dimgray'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # More isometric / seen from above
    ax.view_init(
        elev=35,
        azim=45
    )

    plt.tight_layout()

    plt.savefig(
        "fig_dbscan_before.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()



    ##################################
    # DBSCAN
    ##################################

    labels=DBSCAN(
        eps=0.75,
        min_samples=10
    ).fit_predict(X)



    ##################################
    # FIGURE 2
    ##################################

    cluster_colors=[
        'red',
        'blue',
        'green',
        'orange',
        'purple',
        'cyan'
    ]


    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        projection='3d'
    )


    for i in np.unique(labels):

        mask=labels==i

        if i==-1:

            ax.scatter(
                X[mask,0],
                X[mask,1],
                X[mask,2],
                c='gray',
                s=12
            )

        else:

            ax.scatter(
                X[mask,0],
                X[mask,1],
                X[mask,2],
                c=cluster_colors[
                    i % len(cluster_colors)
                ],
                s=18
            )


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    plt.tight_layout()

    plt.savefig(
        "fig_dbscan_after.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print("Saved DBSCAN figures")


##################################################
# 5 RANSAC CONCEPTS
##################################################

def ransac_figures():

    import matplotlib.pyplot as plt

    np.random.seed(7)

    elev=35
    azim=45


    ########################################
    # 1 PLANE
    ########################################

    x=np.random.uniform(-4,4,500)
    y=np.random.uniform(-4,4,500)

    z=0.5*x + 0.2*y + np.random.normal(
        0,
        0.3,
        500
    )

    noise=np.random.uniform(
        -5,
        5,
        (80,3)
    )

    pts=np.column_stack(
        [x,y,z]
    )

    cloud=np.vstack(
        [pts,noise]
    )


    # BEFORE
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_plane_before.png",
        dpi=300
    )
    plt.close()


    # AFTER (fit plane)
    xx,yy=np.meshgrid(
        np.linspace(-4,4,20),
        np.linspace(-4,4,20)
    )

    zz=0.5*xx + 0.2*yy

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.plot_surface(
        xx,yy,zz,
        alpha=0.5,
        color='red'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_plane_after.png",
        dpi=300
    )
    plt.close()



    ########################################
    # 2 SPHERE
    ########################################

    phi=np.random.uniform(
        0,np.pi,500
    )

    theta=np.random.uniform(
        0,2*np.pi,500
    )

    r=3

    x=r*np.sin(phi)*np.cos(theta)
    y=r*np.sin(phi)*np.sin(theta)
    z=r*np.cos(phi)

    pts=np.column_stack(
        [x,y,z]
    )

    cloud=np.vstack(
        [pts,noise]
    )


    # BEFORE
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_sphere_before.png",
        dpi=300
    )
    plt.close()


    # AFTER
    u,v=np.mgrid[
        0:2*np.pi:20j,
        0:np.pi:20j
    ]

    xs=3*np.cos(u)*np.sin(v)
    ys=3*np.sin(u)*np.sin(v)
    zs=3*np.cos(v)

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.plot_surface(
        xs,ys,zs,
        alpha=0.5,
        color='blue'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_sphere_after.png",
        dpi=300
    )
    plt.close()



    ########################################
    # 3 CYLINDER
    ########################################

    th=np.random.uniform(
        0,
        2*np.pi,
        500
    )

    z=np.random.uniform(
        -4,
        4,
        500
    )

    x=2*np.cos(th)
    y=2*np.sin(th)

    pts=np.column_stack(
        [x,y,z]
    )

    cloud=np.vstack(
        [pts,noise]
    )


    # BEFORE
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_cylinder_before.png",
        dpi=300
    )
    plt.close()


    # AFTER
    th,z=np.meshgrid(
        np.linspace(0,2*np.pi,30),
        np.linspace(-4,4,30)
    )

    xc=2*np.cos(th)
    yc=2*np.sin(th)

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        cloud[:,0],
        cloud[:,1],
        cloud[:,2],
        c='dimgray',
        s=10
    )

    ax.plot_surface(
        xc,yc,z,
        alpha=0.5,
        color='green'
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev,azim=azim)

    plt.tight_layout()
    plt.savefig(
        "fig_ransac_cylinder_after.png",
        dpi=300
    )
    plt.close()


    print("Saved RANSAC figures")


##################################################
# 6 DAMAGE SEGMENTATION / RECONSTRUCTION
##################################################

##################################################
# 6 DAMAGE SEGMENTATION / RECONSTRUCTION
##################################################

def segmentation_reconstruction_figure():

    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    ##################################
    # Load npz files
    ##################################

    damaged=np.load(
        "/home/miguelcruz/Documents/D-PCN2/data/DMC/010/010001.npz"
    )

    recon=np.load(
        "/home/miguelcruz/Documents/D-PCN2/data/DMC/010/010001_gt.npz"
    )

    pts=damaged["points"]
    labels=damaged["labels"]

    recon_pts=recon["points"]
    recon_labels=recon["labels"]


    ##################################
    # Extract labeled points
    ##################################

    damage_pts=pts[
        labels==1
    ]

    rec_pts=recon_pts[
        recon_labels==1
    ]


    ##################################
    # ONLY clean reconstruction points
    # Keep 4 largest clusters
    ##################################

    def keep_top4_clusters(
        points,
        eps=4.8,
        min_samples=25
    ):

        if len(points)==0:
            return points

        labels_db=DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit_predict(points)

        valid=labels_db!=-1

        if np.sum(valid)==0:
            return points

        cluster_ids=np.unique(
            labels_db[valid]
        )

        # sort clusters by size
        sorted_clusters=sorted(
            cluster_ids,
            key=lambda c: np.sum(
                labels_db==c
            ),
            reverse=True
        )

        # keep 4 largest clusters
        keep_clusters=sorted_clusters[:4]

        mask=np.isin(
            labels_db,
            keep_clusters
        )

        return points[
            mask
        ]


    # ONLY reconstruction gets filtered
    rec_pts=keep_top4_clusters(
        rec_pts
    )


    ##################################
    # Downsample for visualization
    ##################################

    np.random.seed(7)

    n_vis=1200
    n_damage=500


    if len(pts)>n_vis:

        idx=np.random.choice(
            len(pts),
            n_vis,
            replace=False
        )

        pts_vis=pts[idx]

    else:

        pts_vis=pts.copy()


    # damage untouched
    if len(damage_pts)>n_damage:

        idx=np.random.choice(
            len(damage_pts),
            n_damage,
            replace=False
        )

        damage_vis=damage_pts[idx]

    else:

        damage_vis=damage_pts.copy()


    if len(rec_pts)>n_damage:

        idx=np.random.choice(
            len(rec_pts),
            n_damage,
            replace=False
        )

        recon_vis=rec_pts[idx]

    else:

        recon_vis=rec_pts.copy()



    ##################################
    # FIGURE 1
    ##################################

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        pts_vis[:,0],
        pts_vis[:,1],
        pts_vis[:,2],
        c='dimgray',
        s=4
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        pts_vis
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_segmentation_input.png",
        dpi=300
    )

    plt.close()



    ##################################
    # FIGURE 2
    ##################################

    all_pts=np.vstack(
        [pts_vis,damage_vis]
    )

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        pts_vis[:,0],
        pts_vis[:,1],
        pts_vis[:,2],
        c='lightgray',
        alpha=0.35,
        s=4
    )

    ax.scatter(
        damage_vis[:,0],
        damage_vis[:,1],
        damage_vis[:,2],
        c='red',
        s=12
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        all_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_segmentation_damage.png",
        dpi=300
    )

    plt.close()



    ##################################
    # FIGURE 3
    ##################################

    all_pts=np.vstack(
        [pts_vis,recon_vis]
    )

    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(projection='3d')

    ax.scatter(
        pts_vis[:,0],
        pts_vis[:,1],
        pts_vis[:,2],
        c='lightgray',
        alpha=0.35,
        s=4
    )

    ax.scatter(
        recon_vis[:,0],
        recon_vis[:,1],
        recon_vis[:,2],
        c='green',
        s=12
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=35,
        azim=45
    )

    set_axes_equal(
        ax,
        all_pts
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    plt.tight_layout()

    plt.savefig(
        "fig_segmentation_reconstruction.png",
        dpi=300
    )

    plt.close()

    print("Saved Figure 6")

##################################################
# EQUAL AXIS SCALING FOR MATPLOTLIB 3D
##################################################

def set_axes_equal(ax, points):

    x=points[:,0]
    y=points[:,1]
    z=points[:,2]

    x_mid=(x.max()+x.min())/2
    y_mid=(y.max()+y.min())/2
    z_mid=(z.max()+z.min())/2

    max_range=max(
        x.max()-x.min(),
        y.max()-y.min(),
        z.max()-z.min()
    )/2

    ax.set_xlim(
        x_mid-max_range,
        x_mid+max_range
    )

    ax.set_ylim(
        y_mid-max_range,
        y_mid+max_range
    )

    ax.set_zlim(
        z_mid-max_range,
        z_mid+max_range
    )

##################################################

if __name__=="__main__":

    # geometry_types()

    # icp_figure()

    # boolean_concept()

    # dbscan_figure()

    # ransac_figures()

    segmentation_reconstruction_figure()
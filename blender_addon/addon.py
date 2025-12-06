bl_info = {
    "name": "Voxelizer Sync Addon",
    "author": "You",
    "version": (0,1),
    "blender": (3,5,0),
    "location": "Scene > Voxelizer",
    "description": "Sync with GitHub and run GPU voxelizer",
    "category": "Object"
}

import bpy, os, subprocess, sys
from bpy.props import StringProperty, FloatProperty

addon_dir = os.path.dirname(__file__)
repo_dir = os.path.abspath(os.path.join(addon_dir, ".."))

class VS_OT_sync_repo(bpy.types.Operator):
    bl_idname = "vs.sync_repo"
    bl_label = "Sync Engine from GitHub"
    def execute(self, context):
        gh_url = context.scene.vs_props.github_url
        dest = os.path.join(repo_dir, "engine_src")
        if not os.path.exists(dest):
            cmd = ["git", "clone", "--depth","1", gh_url, dest]
        else:
            cmd = ["git", "-C", dest, "pull"]
        subprocess.run(cmd, check=False)
        self.report({'INFO'}, "Synced engine to: " + dest)
        return {'FINISHED'}

class VS_OT_run_voxel(bpy.types.Operator):
    bl_idname = "vs.run_voxel"
    bl_label = "Run Voxelizer (Local)"
    def execute(self, context):
        dest = os.path.join(repo_dir, "engine_src")
        pyfile = os.path.join(dest, "run_voxel_cli.py")
        if not os.path.exists(pyfile):
            self.report({'ERROR'}, "Engine not found. Sync first.")
            return {'CANCELLED'}
        temp_ply = os.path.join(dest, "temp_input.ply")
        bpy.ops.export_mesh.ply(filepath=temp_ply, use_normals=True)
        python_exec = context.scene.vs_props.python_exec or sys.executable
        cmd = [python_exec, pyfile, "--input", temp_ply, "--output", os.path.join(dest,"out.npy"), "--pitch", str(context.scene.vs_props.pitch)]
        subprocess.run(cmd, check=True)
        self.report({'INFO'}, "Voxelization finished. Output in engine_src/out.npy")
        return {'FINISHED'}

class VS_PT_panel(bpy.types.Panel):
    bl_label = "Voxelizer"
    bl_idname = "SCENE_PT_voxelizer"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    def draw(self, context):
        layout = self.layout
        props = context.scene.vs_props
        layout.prop(props, "github_url")
        layout.operator("vs.sync_repo")
        layout.prop(props, "python_exec")
        layout.prop(props, "pitch")
        layout.operator("vs.run_voxel")

class VSProps(bpy.types.PropertyGroup):
    github_url: StringProperty(name="GitHub URL", default="https://github.com/yourusername/voxelizer-engine.git")
    python_exec: StringProperty(name="Python Exec", default="")
    pitch: FloatProperty(name="Voxel pitch", default=0.2, min=0.01)

classes = [VS_OT_sync_repo, VS_OT_run_voxel, VS_PT_panel, VSProps]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.vs_props = bpy.props.PointerProperty(type=VSProps)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.vs_props

if __name__ == "__main__":
    register()

# RX 580 Builder test package

The deliverable for this branch is the Windows **Hookshot Builder package**: `HookshotBuilder.exe` plus its sibling `tools/` directory, not a prebuilt game executable.

The Windows workflow downloads the current full Builder tools bundle, overlays this fork's current source, publishes the WPF Builder app, and compiles the changed Vulkan renderer. The resulting `HookshotBuilder.zip` contains the app, full tools bundle and test instructions. It does not contain the user's game files, keys, built game or saves.

## Test steps

1. Extract the complete ZIP into a folder. Keep `tools/` beside `HookshotBuilder.exe`.
2. Run `HookshotBuilder.exe` and build from your own supported game copy.
3. In the folder containing the built `hookshot.exe`, create or edit `hookshot_env.txt` and add:

   ```ini
   HOOKSHOT_NO_UPDATE_CHECK=1
   ```

   Leave `RX_VK_FALLBACK_NO_MESH` unset. Setting it opts out of Vulkan and hands rendering to WebGPU, which does not test the compute fallback.
4. Start the game and test the same menu and scene as before. Send back `hookshot_log.txt` and describe which objects/text render.

## What this build attempts

When Vulkan lacks `VK_EXT_mesh_shader`, eligible mesh programs that have **no task stage** are translated to compute. Direct one-dimensional workgroup grids are dispatched, generated vertices and indirect draw commands are written, and a conventional vertex/fragment pipeline renders the result.

## Important limitations

This is still an early partial fallback, not a complete mesh-shader replacement. Mesh draws with task stages, indirect mesh draws and unsupported output built-ins remain unsupported and may be skipped. Some geometry or visual effects may still be missing. The source has not yet been confirmed on an RX 580; a successful Builder package or renderer compile does not prove that the game renders correctly.

## What to report

- Is the title/menu text readable?
- Are the player, nearby geometry, doors, fences and trees complete?
- Does geometry pop or disappear as you move or change camera distance?
- Does loading a save complete, and does the first playable area render?
- Attach the new `hookshot_log.txt`, including the GPU/driver and mesh-fallback messages.

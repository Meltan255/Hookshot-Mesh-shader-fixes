# RX 580 Builder test package

The deliverable for this branch is the Windows **Hookshot Builder package**: `HookshotBuilder.exe` plus its sibling `tools/` directory, not a prebuilt game executable.

The Windows workflow downloads the current full Builder tools bundle, overlays this fork's current `nx2/`, `host/`, `gpu/`, `third_party/` and `assets/` source, publishes the WPF Builder app, and checks that the changed Vulkan renderer compiles. The resulting `HookshotBuilder.zip` contains the app, full tools bundle, and test instructions. It does not contain the user's game files, keys, built game, or saves.

## Test steps

1. Extract the complete zip into a folder. Keep `tools/` beside `HookshotBuilder.exe`.
2. Run `HookshotBuilder.exe` and build from your own supported game copy.
3. In the folder containing the built `hookshot.exe`, create or edit `hookshot_env.txt` and add:

   ```ini
   RX_VK_FALLBACK_NO_MESH=1
   HOOKSHOT_NO_UPDATE_CHECK=1
   ```

4. Start the game and test the same menu and scene. Send back `hookshot_log.txt` and describe which objects/text render.

The renderer experiment is limited: when Vulkan lacks task/mesh shader support, it declines startup and the host selects the existing WebGPU renderer. WebGPU does not emulate the task/mesh-generated geometry, so this is **not** the complete compute-to-vertex fallback and missing models/foliage may remain. The purpose of this first package is to test the Builder distribution and this renderer hand-off on the RX 580.

//MyGame.fs
namespace MyGame
open System
open Nu
open MenuScreens

type MyGameDispatcher () =
    inherit GameDispatcherImSim ()
    
    override this.Process (_, world) =
        // Fixed virtual resolution - Nu handles all scaling
        Game.SetEye2dSize (v2 640.0f 360.0f) world
        //Game.SetEye2dCenter v3Zero world
        MenuScreens.MainMenuScreen.render world 
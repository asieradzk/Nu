﻿namespace TerraFirma
open System
open Nu
open TerraFirma

// this is a plugin for the Nu game engine that directs the execution of your application and editor
type TerraFirmaPlugin () =
    inherit NuPlugin ()

    // this exposes different editing modes in the editor
    override this.EditModes =
        Map.ofList
            [("Splash", fun world -> Game.SetGameState Splash world)
             ("Title", fun world -> Game.SetGameState Title world)
             ("Credits", fun world -> Game.SetGameState Credits world)
             ("Gameplay", fun world ->
                Simulants.Gameplay.SetGameplayState Playing world
                Game.SetGameState Gameplay world)]

    // this specifies which packages are automatically loaded at game start-up.
    override this.InitialPackages =
        [Assets.Gui.PackageName]
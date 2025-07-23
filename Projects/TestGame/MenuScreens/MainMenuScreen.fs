namespace MenuScreens
open Nu

module MainMenuScreen =

    let render world =
        let _ = World.beginScreen "MainMenu" true Vanilla [] world
        World.beginGroup "UI" [] world
        
        AnimatedBackground.render world

        World.endGroup world
        World.endScreen world
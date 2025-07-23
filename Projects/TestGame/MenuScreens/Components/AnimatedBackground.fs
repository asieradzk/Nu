module AnimatedBackground

open Nu
open System.Numerics
open MyGame


let SkyImage : Image AssetTag = { PackageName = "MyGame"; AssetName = "cartoonsky" }
let CloudImage : Image AssetTag = { PackageName = "MyGame"; AssetName = "cartooncloudsmall" }
let CloudSpeed = 30f

let CloudScale = 0.3f
let CloudOpacity = 0.7f


let render world =
    let virtualSize = Game.GetEye2dSize world
    let time = world.GameTime.Seconds

   
    World.beginPanel "SkyPanel" [
        Entity.BackdropImageOpt .= Some SkyImage
        Entity.Position .= v3 0.0f 0.0f 0.0f
        Entity.Size .= v3 virtualSize.X virtualSize.Y 0.0f
        Entity.ElevationLocal .= 0.0f
        Entity.Layout .= Manual 
    ] world

    let cloudWidth = 100.0f
    // Uses modulo to loop the cloud position: as (time * speed) 
    //increases, % wraps it within 0 to (screen width + cloud width), 
    //creating seamless off-screen reset without if/else checks. Subtractions center it initially off-left.
    let positionX = (time * CloudSpeed) % (virtualSize.X + cloudWidth) - virtualSize.X * 0.5f - cloudWidth * 0.5f 


    let colorWithOpacity = Color.One.WithA CloudOpacity
    World.doStaticSprite "CloudSprite" [
        Entity.StaticImage .= CloudImage
        Entity.Position @= v3 positionX 65f 0.0f
        Entity.Scale .= v3 CloudScale CloudScale CloudScale
        Entity.Size .= v3 256f 256f 0f
        Entity.Color .= colorWithOpacity
        Entity.ElevationLocal .= 1.0f    
    ] world

    RenderGraphHelpers.doBlackSquare "TestSquare" [
        Entity.Position .= v3 0.0f 0.0f 0.0f  // Center of screen
        Entity.Size .= v3 200.0f 200.0f 0.0f
        Entity.ElevationLocal .= 5.0f
    ] world


    World.endPanel world



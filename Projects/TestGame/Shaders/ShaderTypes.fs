module MyGame.ShaderTypes

open Nu
open RenderGraph

//concrete type for shader storage to avoid type inference issues
//Here only temporarily, belongs in render graph or somewhere
type ShaderPackage = 
    { ShaderInstance : obj
      ShaderProperties : obj 
      //Pre-built function that extracts everything we needs
      //to avoid runtime reflection
      ExtractData : unit -> (ShaderSource * (string * UniformValue) array) }
    
    static member Default = 
        { ShaderInstance = null
          ShaderProperties = null
          ExtractData = fun () -> (ShaderSource.SeparateFiles("", ""), [||]) }
namespace MyGame

open System
open System.Numerics
open Prime
open Nu
open MyGame.ShaderTypes  // For ShaderPackage type

//properties for TestBasic shader
//you always need to define them 
//ideally they should come from a pool of allowed ones like/ single, texture, etc and give compiler errors if used incorrectly

module TestBasicShaderTypes =
    [<Struct>]
    type Properties = {
        uColor: Color
        uTime: single 
        uSpeed: single
    }

//user defined shader access - can be code generated 
//or user can write their custom logic
type TestBasicShader() =
    inherit Nu.Shader<TestBasicShaderTypes.Properties>()
    
    //instance for caching
    static let instance = TestBasicShader()
    
    //source specification
    //we expect a wild variety not to mention per-pipeline. Good idea to have sensible defaults while allowing users flexible modification if they desire
    override _.Source = 
        Nu.ShaderSource.SeparateFiles("Assets/Shaders/basicv2.vert", "Assets/Shaders/basicv2.frag")
    
    //creation function curried for functional style\
    //can aslo be source genrated, users can omit, write different style if they want
    static member create = fun color time speed ->
        let props = { TestBasicShaderTypes.Properties.uColor = color; TestBasicShaderTypes.Properties.uTime = time; TestBasicShaderTypes.Properties.uSpeed = speed }
        { ShaderInstance = box instance
          ShaderProperties = box props
          // Closure captures everything - no reflection needed at runtime!
          ExtractData = fun () -> (instance.Source, instance.ExtractUniforms(props)) }
    
    //example alternative create method someone might want to use
    static member Create(color: Color, time: single, speed: single) =
        TestBasicShader.create color time speed
    
    //convenience functions - something like this can serve as replacement for material system
    static member black = TestBasicShader.create Color.Black 0.0f 1.0f
    static member red = TestBasicShader.create Color.DarkRed 0.0f 1.0f  
    static member white = TestBasicShader.create Color.White 0.0f 1.0f
    static member animated time speed = TestBasicShader.create Color.White time speed
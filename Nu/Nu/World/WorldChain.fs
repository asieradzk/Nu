﻿// Nu Game Engine.
// Copyright (C) Bryan Edds.

namespace Nu
open System
open System.Diagnostics
open Prime

/// The Chain monad. Allows the user to define a chain of operations over the world that
/// optionally spans across a bounded number of events.
type [<ReferenceEquality>] Chain<'e, 'a> =
    Chain of (World -> Either<'e -> Chain<'e, 'a>, 'a>)

/// Implements the chain monad.
type ChainBuilder () =

    /// Functor map for the chain monad.
    [<DebuggerHidden; DebuggerStepThrough>]
    member this.Map (f : 'a -> 'b) (a : Chain<'e, 'a>) : Chain<'e, 'b> =
        Chain (fun world ->
            let chainMapper eir =
                match eir with
                | Left c -> Left (fun world -> this.Map f (c world))
                | Right a -> Right (f a)
            let eir = match a with Chain b -> b world
            chainMapper eir)

    /// Applicative apply for the chain monad.
    /// TODO: Implement!
    [<DebuggerHidden; DebuggerStepThrough>]
    member this.Apply (c : Chain<'e, 'a -> 'b>) (_ : Chain<'e, 'a>) : Chain<'e, 'b> =
        Chain (fun world ->
            match (match c with Chain f -> f world) with
            //                             ^--- NOTE: unbounded recursion here
            | _ -> failwithnie ())

    /// Monadic return for the chain monad.
    [<DebuggerHidden; DebuggerStepThrough>]
    member this.Return (a : 'a) : Chain<'e, 'a> =
        Chain (fun _ -> Right a)

    /// Monadic bind for the chain monad.
    [<DebuggerHidden; DebuggerStepThrough>]
    member this.Bind (c : Chain<'e, 'a>, cont : 'a -> Chain<'e, 'b>) : Chain<'e, 'b> =
        Chain (fun world ->
            match (match c with Chain f -> f world) with
            //                             ^--- NOTE: unbounded recursion here
            | Left c -> Left (fun e -> this.Bind (c e, cont))
            | Right v -> match cont v with Chain f -> f world)

[<AutoOpen>]
module ChainBuilder =

    /// Builds chains.
    let [<DebuggerHidden>] chain = ChainBuilder ()

/// Chain operators.
module Chain =

    /// Functor map for the chain monad.
    let [<DebuggerHidden; DebuggerStepThrough>] inline map f a = chain.Map f a

    /// Functor map for the chain monad.
    let [<DebuggerHidden; DebuggerStepThrough>] inline apply c a = chain.Apply c a

    /// Monadic return for the chain monad.
    let [<DebuggerHidden; DebuggerStepThrough>] inline returnM a = chain.Return a

    /// Monadic bind for the chain monad.
    let [<DebuggerHidden; DebuggerStepThrough>] inline bind c a = chain.Bind (c, a)

    /// Get the world.
    let [<DebuggerHidden>] get : Chain<'e, World> =
        Chain (fun world -> Right world)

    /// Get the world as transformed via 'by'.
    let [<DebuggerHidden; DebuggerStepThrough>] getBy by : Chain<'e, 'a> =
        Chain (fun world -> Right (by world))

    /// Update the world with an additional transformed world parameter.
    let [<DebuggerHidden; DebuggerStepThrough>] updateBy by expr : Chain<'e, unit> =
        Chain (fun world -> expr (by world) world; Right ())

    /// Update the world.
    let [<DebuggerHidden; DebuggerStepThrough>] update expr : Chain<'e, unit> =
        Chain (fun world -> expr world; Right ())

    /// Get the next event.
    let [<DebuggerHidden>] next : Chain<'e, 'e> =
        Chain (fun _ -> Left returnM)

    /// Pass over the next event.
    let [<DebuggerHidden>] pass : Chain<'e, unit> =
        Chain (fun _ -> Left (fun _ -> returnM ()))

    /// React to the next event, using the event's data in the reaction.
    let [<DebuggerHidden; DebuggerStepThrough>] reactData<'a, 's when 's :> Simulant> expr : Chain<Event<'a, 's>, unit> =
       chain {
           let! e = next
           let! world = get
           expr (e.Data) world
           return () }

    /// React to the next event, using the event's value in the reaction.
    let [<DebuggerHidden; DebuggerStepThrough>] reactEvent expr : Chain<'e, unit> =
        chain {
            let! e = next
            let! world = get
            expr e world
            return () }

    /// React to the next event, discarding the event's value.
    let [<DebuggerHidden; DebuggerStepThrough>] react expr : Chain<'e, unit> =
        chain {
            do! pass
            let! world = get
            expr world
            return () }

    /// Loop in a chain context while 'pred' evaluate to true considering only the loop data.
    let rec [<DebuggerHidden; DebuggerStepThrough>] loop (i : 'i) (step : 'i -> 'i) (pred : 'i -> World -> bool) (c : 'i -> Chain<'e, unit>) =
        chain {
            let! world = get
            do! if pred i world then
                    chain {
                        do! c i
                        let i = step i
                        do! loop i step pred c }
                else returnM () }

    /// Loop in a chain context while 'pred' evaluates to true considering only the world state.
    let [<DebuggerHidden; DebuggerStepThrough>] during (pred : World -> bool) (c : Chain<'e, unit>) =
        loop () id (fun _ -> pred) (fun _ -> c)

    /// Step once into a chain.
    let [<DebuggerHidden; DebuggerStepThrough>] step (c : Chain<'e, 'a>) (world : World) : Either<'e -> Chain<'e, 'a>, 'a> =
        match c with Chain f -> f world

    /// Advance a chain value by one step, providing 'e'.
    let [<DebuggerHidden; DebuggerStepThrough>] advance (c : 'e -> Chain<'e, 'a>) (e : 'e) (world : World) : Either<'e -> Chain<'e, 'a>, 'a> =
        step (c e) world

[<AutoOpen>]
module WorldChain =

    type World with

        /// Run a chain to its end, providing 'e' for all its steps.
        [<DebuggerHidden; DebuggerStepThrough>] 
        static member chainConstant (c : Chain<'e, 'a>) (e : 'e) (world : World) : 'a =
            match Chain.step c world with
            | Left chain -> World.chainConstant (chain e) e world
            | Right v -> v

        /// Run a chain to its end, providing unit for all its steps.
        [<DebuggerHidden; DebuggerStepThrough>]
        static member chainUnit (c : Chain<unit, 'a>) (world : World) : 'a =
            World.chainConstant c () world
            
        /// Execute a chain over the given stream.
        [<DebuggerHidden; DebuggerStepThrough>]
        static member chainPlus (c : Chain<Event<'a, Simulant>, unit>) (stream : Stream<'a>) (handling : Handling) (world : World) =
            let globalSimulant = World.getGlobalSimulantGeneralized world
            let stateId = Gen.id64
            let subscriptionId = Gen.id64
            World.addEventState stateId (fun (_ : Event<'a, Simulant>) -> c) world
            let (eventAddress, unsubscribe) = stream.Subscribe world
            let unsubscribe = fun world ->
                World.removeEventState stateId world
                unsubscribe world
                World.unsubscribe subscriptionId world
            let advance = fun evt world ->
                let chain = World.getEventState stateId world : Event<'a, Simulant> -> Chain<Event<'a, Simulant>, unit>
                let advanceResult = Chain.advance chain evt world
                match advanceResult with
                | Right () -> unsubscribe world
                | Left chainNext -> World.addEventState stateId chainNext world
            let callback = fun evt world ->
                advance evt world
                handling
            advance Unchecked.defaultof<Event<'a, Simulant>> world
            World.subscribePlus<'a, Simulant> subscriptionId callback eventAddress globalSimulant world |> ignore
            unsubscribe

        /// Execute a chain over the given stream.
        /// Allows each chained operation to run without referencing its source event, and without specifying its event
        /// handling approach by assuming Cascade.
        [<DebuggerHidden; DebuggerStepThrough>]
        static member chain (c : Chain<Event<'a, Simulant>, unit>) (stream : Stream<'a>) world =
            World.chainPlus c stream Cascade world |> ignore
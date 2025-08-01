﻿// Nu Game Engine.
// Copyright (C) Bryan Edds.

namespace Nu
open System
open System.Collections.Generic
open System.Numerics
open Prime

/// Masks for Octelement flags.
module OctelementMasks =

    // OPTIMIZATION: Octelement flag bit-masks for performance.
    let [<Literal>] VisibleInViewMask = 0b0001u
    let [<Literal>] StaticInPlayMask =  0b0010u
    let [<Literal>] LightProbeMask =    0b0100u
    let [<Literal>] LightMask =         0b1000u

// NOTE: opening this in order to make the Octelement property implementations reasonably succinct.
open OctelementMasks

[<RequireQualifiedAccess>]
module Octelement =

    /// An element in an octree.
    /// NOTE: we intentionally use incomplete equality semantics here so these can be stored in a HashSet.
    type [<CustomEquality; NoComparison; Struct>] Octelement<'e when 'e : equality> =
        private
            { HashCode_ : int // OPTIMIZATION: cache hash code to increase look-up speed.
              Flags_ : uint
              Presence_ : Presence
              PresenceInPlay_ : Presence
              Bounds_ : Box3
              Entry_ : 'e }
        member this.VisibleInView = this.Flags_ &&& VisibleInViewMask <> 0u
        member this.StaticInPlay = this.Flags_ &&& StaticInPlayMask <> 0u
        member this.LightProbe = this.Flags_ &&& LightProbeMask <> 0u
        member this.Light = this.Flags_ &&& LightMask <> 0u
        member this.Presence = this.Presence_
        member this.PresenceInPlay = this.PresenceInPlay_
        member this.Bounds = this.Bounds_
        member this.Entry = this.Entry_
        override this.GetHashCode () = this.HashCode_
        override this.Equals that = match that with :? Octelement<'e> as that -> this.Entry_.Equals that.Entry_ | _ -> false

    let make visibleInView staticInPlay lightProbe light presence presenceInPlay bounds (entry : 'e) =
        let hashCode = entry.GetHashCode ()
        let flags =
            (if visibleInView then VisibleInViewMask else 0u) |||
            (if staticInPlay then StaticInPlayMask else 0u) |||
            (if lightProbe then LightProbeMask else 0u) |||
            (if light then LightMask else 0u)
        { HashCode_ = hashCode; Flags_ = flags; Presence_ = presence; PresenceInPlay_ = presenceInPlay; Bounds_ = bounds; Entry_ = entry }

/// An element in an octree.
type Octelement<'e when 'e : equality> = Octelement.Octelement<'e>

/// Equality compares two octelements.
type OctelementEqualityComparer<'e when 'e : equality> () =
    interface 'e Octelement IEqualityComparer with
        member this.Equals (left, right) = left.Entry = right.Entry // OPTIMIZATION: inline equality to avoid allocation.
        member this.GetHashCode element = element.GetHashCode ()

[<RequireQualifiedAccess>]
module internal Octnode =

    type [<Struct>] internal Octchildren<'e when 'e : equality> =
        | NoChildren
        | NodeChildren of NodeChildren : 'e Octnode array
        | ElementChildren of ElementChildren : 'e Octelement HashSet

    and internal Octnode<'e when 'e : equality> =
        private
            { mutable ElementsCount_ : int // OPTIMIZATION: keeps track of total contained elements in order to get an early-out on queries.
              Id_ : uint64
              Depth_ : int
              Bounds_ : Box3
              mutable Children_ : 'e Octchildren
              Comparer_ : 'e OctelementEqualityComparer
              Leaves_ : Dictionary<Vector3, 'e Octnode> }

        override this.ToString () = ""
        member this.Id = this.Id_

    let internal makeChildren<'e when 'e : equality> node =
        let childSize = node.Bounds_.Size * 0.5f
        let childDepth = dec node.Depth_
        if childDepth > 0 then
            let nodeChildren =
                [|for i in 0 .. 1 do
                    [|for j in 0 .. 1 do
                        [|for k in 0 .. 1 do
                            let childOffset = v3 (childSize.X * single i) (childSize.Y * single j) (childSize.Z * single k)
                            let childMin = node.Bounds_.Min + childOffset
                            let childBounds = box3 childMin childSize
                            let child =
                                { ElementsCount_ = 0
                                  Id_ = Gen.id64
                                  Depth_ = childDepth
                                  Bounds_ = childBounds
                                  Children_ = NoChildren
                                  Comparer_ = node.Comparer_
                                  Leaves_ = node.Leaves_ }
                            if childDepth = 1 then node.Leaves_.Add (childBounds.Min, child)
                            child|]|]|]
            NodeChildren (nodeChildren |> Array.concat |> Array.concat)
        else
            let children = HashSet<'e Octelement> node.Comparer_
            ElementChildren children

    let internal atPoint (point : Vector3) (node : 'e Octnode) =
        node.Bounds_.Intersects point

    let internal isIntersectingBox (bounds : Box3) (node : 'e Octnode) =
        node.Bounds_.Intersects bounds

    let inline internal isIntersectingFrustum (frustum : Frustum) (node : 'e Octnode) =
        frustum.Intersects node.Bounds_

    let inline internal containsBoxExclusive (bounds : Box3) (node : 'e Octnode) =
        node.Bounds_.ContainsExclusive bounds = ContainmentType.Contains

    let rec internal addElement bounds (element : 'e Octelement inref) (node : 'e Octnode) : int =
        let delta =
            if isIntersectingBox bounds node then
                match node.Children_ with
                | NoChildren ->
                    node.Children_ <- makeChildren node
                    addElement bounds &element node
                | NodeChildren nodes ->
                    let mutable delta = 0
                    for i in 0 .. dec nodes.Length do
                        let node = nodes.[i]
                        delta <- delta + addElement bounds &element node
                    delta
                | ElementChildren elements ->
                    let removed = elements.Remove element
                    let added = elements.Add element
                    if removed
                    then (if not added then -1 else 0)
                    else (if not added then 0 else 1)
            else 0
        node.ElementsCount_ <- node.ElementsCount_ + delta
        delta

    let rec internal removeElement bounds (element : 'e Octelement inref) (node : 'e Octnode) =
        let delta =
            if isIntersectingBox bounds node then
                match node.Children_ with
                | NoChildren ->
                    0
                | NodeChildren nodes ->
                    let mutable delta = 0
                    for i in 0 .. dec nodes.Length do
                        let node = nodes.[i]
                        delta <- delta + removeElement bounds &element node
                    delta
                | ElementChildren elements ->
                    if elements.Remove element then -1 else 0
            else 0
        node.ElementsCount_ <- node.ElementsCount_ + delta
        delta

    let rec internal updateElement boundsOld boundsNew (element : 'e Octelement inref) (node : 'e Octnode) =
        let delta =
            match node.Children_ with
            | NoChildren ->
                if isIntersectingBox boundsOld node || isIntersectingBox boundsNew node then
                    node.Children_ <- makeChildren node
                    updateElement boundsOld boundsNew &element node
                else 0
            | NodeChildren nodes ->
                let mutable delta = 0
                for i in 0 .. dec nodes.Length do
                    let node = nodes.[i]
                    if isIntersectingBox boundsOld node || isIntersectingBox boundsNew node then
                        delta <- delta + updateElement boundsOld boundsNew &element node
                delta
            | ElementChildren elements ->
                if isIntersectingBox boundsNew node then
                    let removed = elements.Remove element
                    let added = elements.Add element
                    if removed
                    then (if not added then -1 else 0)
                    else (if not added then 0 else 1)
                elif isIntersectingBox boundsOld node then
                    if elements.Remove element then -1 else 0
                else 0
        node.ElementsCount_ <- node.ElementsCount_ + delta
        delta

    let rec internal clearElements node =
        node.ElementsCount_ <- 0
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = &nodes.[i]
                clearElements node
        | ElementChildren children ->
            children.Clear ()

    let rec internal getElementsAtPoint point (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && atPoint point node then
                    getElementsAtPoint point set node
        | ElementChildren elements ->
            for element in elements do
                let bounds = element.Bounds
                if bounds.Intersects point then
                    set.Add element |> ignore

    let rec internal getElementsInBox box (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingBox box node then
                    getElementsInBox box set node
        | ElementChildren elements ->
            for element in elements do
                let bounds = element.Bounds
                if bounds.Intersects box then
                    set.Add element |> ignore

    let rec internal getElementsInFrustum frustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingFrustum frustum node then
                    getElementsInFrustum frustum set node
        | ElementChildren elements ->
            for element in elements do
                let bounds = element.Bounds
                if frustum.Intersects bounds then
                    set.Add element |> ignore

    let rec internal getElementsInPlayBox box (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingBox box node then
                    getElementsInPlayBox box set node
        | ElementChildren elements ->
            for element in elements do
                let presence = element.Presence
                if not element.StaticInPlay && not presence.IsImposter && not presence.IsOmnipresent then
                    let bounds = element.Bounds
                    if bounds.Intersects box then
                        set.Add element |> ignore

    let rec internal getLightProbesInViewFrustum frustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingFrustum frustum node then
                    getLightProbesInViewFrustum frustum set node
        | ElementChildren elements ->
            for element in elements do
                if element.LightProbe && element.VisibleInView then
                    let bounds = element.Bounds
                    if frustum.Intersects bounds then
                        set.Add element |> ignore

    let rec internal getLightProbesInViewBox box (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingBox box node then
                    getLightProbesInViewBox box set node
        | ElementChildren elements ->
            for element in elements do
                if element.LightProbe && element.VisibleInView then
                    let bounds = element.Bounds
                    if box.Intersects bounds then
                        set.Add element |> ignore

    let rec internal getLightProbes (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 then
                    getLightProbes set node
        | ElementChildren elements ->
            for element in elements do
                if element.LightProbe && element.VisibleInView then
                    set.Add element |> ignore

    let rec internal getLightsInViewFrustum frustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingFrustum frustum node then
                    getLightsInViewFrustum frustum set node
        | ElementChildren elements ->
            for element in elements do
                if element.Light && element.VisibleInView then
                    let bounds = element.Bounds
                    if frustum.Intersects bounds then
                        set.Add element |> ignore

    let rec internal getLightsInViewBox box (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingBox box node then
                    getLightsInViewBox box set node
        | ElementChildren elements ->
            for element in elements do
                if element.Light && element.VisibleInView then
                    let bounds = element.Bounds
                    if bounds.Intersects box then
                        set.Add element |> ignore

    let rec internal getElementsInPlayFrustum frustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingFrustum frustum node then
                    getElementsInPlayFrustum frustum set node
        | ElementChildren elements ->
            for element in elements do
                let presence = element.Presence
                if not element.StaticInPlay && not presence.IsImposter && not presence.IsOmnipresent then
                    let bounds = element.Bounds
                    if frustum.Intersects bounds then
                        set.Add element |> ignore

    let rec internal getElementsInViewFrustum interior exterior frustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingFrustum frustum node then
                    getElementsInViewFrustum interior exterior frustum set node
        | ElementChildren elements ->
            for element in elements do
                let presence = element.Presence
                if interior then
                    if presence.IsInterior || presence.IsExterior then
                        if element.VisibleInView && frustum.Intersects element.Bounds then
                            set.Add element |> ignore
                elif exterior then
                    if presence.IsExterior then
                        if element.VisibleInView && frustum.Intersects element.Bounds then
                            set.Add element |> ignore

    let rec internal getElementsInViewBox box (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 && isIntersectingBox box node then
                    getElementsInViewBox box set node
        | ElementChildren elements ->
            for element in elements do
                if element.VisibleInView && box.Intersects element.Bounds then
                    set.Add element |> ignore

    let rec internal getElementsInView frustumInterior frustumExterior lightBox (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 then
                    let intersectingInterior = isIntersectingFrustum frustumInterior node
                    let intersectingExterior = isIntersectingFrustum frustumExterior node
                    if intersectingInterior || intersectingExterior then
                        if intersectingInterior then getElementsInViewFrustum true false frustumInterior set node
                        if intersectingExterior then getElementsInViewFrustum false true frustumExterior set node
                    if isIntersectingBox lightBox node then
                        getLightsInViewBox lightBox set node
        | ElementChildren _ -> ()

    let rec internal getElementsInPlay playBox playFrustum (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 then
                    if isIntersectingBox playBox node then
                        getElementsInPlayBox playBox set node
                    if isIntersectingFrustum playFrustum node then
                        getElementsInPlayFrustum playFrustum set node
        | ElementChildren _ -> ()

    let rec internal getElements (set : 'e Octelement HashSet) (node : 'e Octnode) =
        match node.Children_ with
        | NoChildren ->
            ()
        | NodeChildren nodes ->
            for i in 0 .. dec nodes.Length do
                let node = nodes.[i]
                if node.ElementsCount_ > 0 then
                    getElements set node
        | ElementChildren children ->
            set.UnionWith children

    let rec internal sweep (node : 'e Octnode) =
        if node.ElementsCount_ = 0 then
            match node.Children_ with
            | NoChildren ->
                ()
            | NodeChildren nodes ->
                for i in 0 .. dec nodes.Length do
                    let node = &nodes.[i]
                    sweep node
            | ElementChildren _ ->
                node.Leaves_.Remove node.Bounds_.Min |> ignore<bool>
            node.Children_ <- NoChildren

    let internal make<'e when 'e : equality> comparer depth (bounds : Box3) (leaves : Dictionary<Vector3, 'e Octnode>) : 'e Octnode =
        if depth < 1 then failwith "Invalid depth for Octnode. Expected value of at least 1."
        let node =
            { ElementsCount_ = 0
              Id_ = Gen.id64
              Depth_ = depth
              Bounds_ = bounds
              Children_ = NoChildren
              Comparer_ = comparer
              Leaves_ = leaves }
        node

type internal Octnode<'e when 'e : equality> = Octnode.Octnode<'e>

[<RequireQualifiedAccess>]
module Octree =

    /// A spatial structure that organizes elements in a 3d grid.
    type [<ReferenceEquality>] Octree<'e when 'e : equality> =
        private
            { LeafSize : Vector3 // TODO: consider keeping the inverse of this to avoid divides.
              Leaves : Dictionary<Vector3, 'e Octnode>
              Imposter : 'e Octelement HashSet
              Omnipresent : 'e Octelement HashSet
              OmnipresentInPlayOnly : 'e Octelement HashSet
              UbiquitousFallback : 'e Octelement HashSet
              Node : 'e Octnode
              Depth : int
              Bounds : Box3 }

    let private tryFindLeafFast (bounds : Box3) tree : 'e Octnode option =
        let offset = -tree.Bounds.Min // use offset to bring div ops into positive space
        let divs = (bounds.Min + offset) / tree.LeafSize
        let evens = v3 (divs.X |> int |> single) (divs.Y |> int |> single) (divs.Z |> int |> single)
        let leafKey = evens * tree.LeafSize - offset
        match tree.Leaves.TryGetValue leafKey with
        | (true, leaf) when Octnode.containsBoxExclusive bounds leaf -> Some leaf
        | (_, _) -> None

    /// Add an element with the given presence and bounds to the tree.
    let addElement (presence : Presence) (presenceInPlay : Presence) bounds (element : 'e Octelement) tree =

        // add to imposter when such
        if presence.IsImposter then
            tree.Imposter.Add element |> ignore

        // add to omnipresent when such
        if presence.IsOmnipresent then
            tree.Omnipresent.Add element |> ignore

        // add to omnipresent-in-play-only when appropriate
        let omnipresentInPlayOnly = not presence.IsOmnipresent && presenceInPlay.IsOmnipresent
        if omnipresentInPlayOnly then
            tree.OmnipresentInPlayOnly.Remove element |> ignore
            tree.OmnipresentInPlayOnly.Add element |> ignore

        // add to node tree or ubiquitous fallback
        if  not (Octnode.isIntersectingBox bounds tree.Node) ||
            bounds.Size.Magnitude >= Constants.Engine.OctreeElementMagnitudeMax then
            tree.UbiquitousFallback.Remove element |> ignore
            tree.UbiquitousFallback.Add element |> ignore
        else Octnode.addElement bounds &element tree.Node |> ignore

    /// Remove an element with the given presence and bounds from the tree.
    let removeElement (presence : Presence) (presenceInPlay : Presence) bounds (element : 'e Octelement) tree =

        // remove from imposter when appropriate
        if presence.IsImposter then
            tree.Imposter.Remove element |> ignore

        // remove from omnipresent when appropriate
        if presence.IsOmnipresent then
            tree.Omnipresent.Remove element |> ignore

        // remove from omnipresent-in-play-only when appropriate
        let omnipresentInPlayOnly = not presence.IsOmnipresent && presenceInPlay.IsOmnipresent
        if omnipresentInPlayOnly then
            tree.OmnipresentInPlayOnly.Remove element |> ignore

        // remove from node tree or ubiquitous fallback
        if  not (Octnode.isIntersectingBox bounds tree.Node) ||
            bounds.Size.Magnitude >= Constants.Engine.OctreeElementMagnitudeMax then
            tree.UbiquitousFallback.Remove element |> ignore
        else Octnode.removeElement bounds &element tree.Node |> ignore

    /// Update an existing element in the tree.
    let updateElement (presenceOld : Presence) (presenceInPlayOld : Presence) boundsOld (presenceNew : Presence) (presenceInPlayNew : Presence) boundsNew element tree =

        // update imposter where appropriate
        if presenceOld.IsImposter then tree.Omnipresent.Remove element |> ignore
        if presenceNew.IsImposter then tree.Omnipresent.Add element |> ignore

        // update omnipresent where appropriate
        if presenceOld.IsOmnipresent then tree.OmnipresentInPlayOnly.Remove element |> ignore
        if presenceNew.IsOmnipresent then tree.OmnipresentInPlayOnly.Add element |> ignore

        // update omnipresent-in-play-only where appropriate
        let omnipresentInPlayOnlyOld = not presenceOld.IsOmnipresent && presenceInPlayOld.IsOmnipresent
        let omnipresentInPlayOnlyNew = not presenceNew.IsOmnipresent && presenceInPlayNew.IsOmnipresent
        if omnipresentInPlayOnlyOld then tree.OmnipresentInPlayOnly.Remove element |> ignore
        if omnipresentInPlayOnlyNew then tree.OmnipresentInPlayOnly.Add element |> ignore

        // update in node tree or ubiquitous fallback
        let wasInNode = Octnode.isIntersectingBox boundsOld tree.Node && boundsOld.Size.Magnitude < Constants.Engine.OctreeElementMagnitudeMax
        let isInNode = Octnode.isIntersectingBox boundsNew tree.Node && boundsNew.Size.Magnitude < Constants.Engine.OctreeElementMagnitudeMax
        if wasInNode then
            if isInNode then
                match tryFindLeafFast boundsOld tree with
                | Some leafOld ->
                    match tryFindLeafFast boundsNew tree with
                    | Some leafNew ->
                        if leafOld.Id = leafNew.Id
                        then Octnode.updateElement boundsOld boundsNew &element leafNew |> ignore
                        else Octnode.updateElement boundsOld boundsNew &element tree.Node |> ignore
                    | None -> Octnode.updateElement boundsOld boundsNew &element tree.Node |> ignore
                | None -> Octnode.updateElement boundsOld boundsNew &element tree.Node |> ignore
            else
                tree.UbiquitousFallback.Remove element |> ignore
                tree.UbiquitousFallback.Add element |> ignore
                Octnode.removeElement boundsOld &element tree.Node |> ignore
        else
            if isInNode then
                tree.UbiquitousFallback.Remove element |> ignore
                Octnode.addElement boundsNew &element tree.Node |> ignore
            else
                tree.UbiquitousFallback.Remove element |> ignore
                tree.UbiquitousFallback.Add element |> ignore

    /// Clear the contents of the tree.
    let clear tree =
        tree.Imposter.Clear ()
        tree.Omnipresent.Clear ()
        tree.OmnipresentInPlayOnly.Clear ()
        tree.UbiquitousFallback.Clear ()
        Octnode.clearElements tree.Node

    /// Get all of the elements in a tree that are in a node intersected by the given point.
    let getElementsAtPoint (point : Vector3) (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            let bounds = imposter.Bounds
            if bounds.Intersects point then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsAtPoint point set tree.Node

    /// Get all of the elements in a tree that are in a node intersected by the given bounds.
    let getElementsInBounds (bounds : Box3) (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            if bounds.Intersects imposter.Bounds then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInBox bounds set tree.Node

    /// Get all of the elements in a tree that are in a node intersected by the given frustum.
    let getElementsInFrustum (frustum : Frustum) (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            if frustum.Intersects imposter.Bounds then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInFrustum frustum set tree.Node

    /// Get all of the elements in a tree.
    let getElements (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElements set tree.Node

    /// Get all of the elements in a tree that satisfy the given query parameters.
    let getElementsInViewFrustum interior exterior (frustum : Frustum) (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            if frustum.Intersects imposter.Bounds then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInViewFrustum interior exterior frustum set tree.Node

    /// Get all of the elements in a tree that satisfy the given query parameters.
    let getElementsInViewBox (box : Box3) (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            if box.Intersects imposter.Bounds then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInViewBox box set tree.Node

    /// Get all of the elements in a tree that are in a node intersected by one of the given frustums or light box depending on its attributes.
    let getElementsInView frustumInterior frustumExterior (frustumImposter : Frustum) lightBox (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            if frustumImposter.Intersects imposter.Bounds then
                set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInView frustumInterior frustumExterior lightBox set tree.Node

    /// Get all of the light probe elements in the given frustum.
    let getLightProbesInViewFrustum frustum (set : _ HashSet) tree =
        for omnipresent in tree.Omnipresent do
            if omnipresent.LightProbe && omnipresent.VisibleInView then
                set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            if ubiquitous.LightProbe && ubiquitous.VisibleInView then
                set.Add ubiquitous |> ignore<bool>
        Octnode.getLightProbesInViewFrustum frustum set tree.Node

    /// Get all of the light probe elements in the given box.
    let getLightProbesInViewBox box (set : _ HashSet) tree =
        for omnipresent in tree.Omnipresent do
            if omnipresent.LightProbe && omnipresent.VisibleInView then
                set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            if ubiquitous.LightProbe && ubiquitous.VisibleInView then
                set.Add ubiquitous |> ignore<bool>
        Octnode.getLightProbesInViewBox box set tree.Node

    /// Get all of the light probe elements.
    let getLightProbesInView (set : _ HashSet) tree =
        for omnipresent in tree.Omnipresent do
            if omnipresent.LightProbe && omnipresent.VisibleInView then
                set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            if ubiquitous.LightProbe && ubiquitous.VisibleInView then
                set.Add ubiquitous |> ignore<bool>
        Octnode.getLightProbes set tree.Node

    /// Get all of the light elements in the given frustum.
    let getLightsInViewFrustum frustum (set : _ HashSet) tree =
        for omnipresent in tree.Omnipresent do
            if omnipresent.Light && omnipresent.VisibleInView then
                set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            if ubiquitous.Light && ubiquitous.VisibleInView then
                set.Add ubiquitous |> ignore<bool>
        Octnode.getLightsInViewFrustum frustum set tree.Node

    /// Get all of the light elements in the given box.
    let getLightsInViewBox box (set : _ HashSet) tree =
        for omnipresent in tree.Omnipresent do
            if omnipresent.Light && omnipresent.VisibleInView then
                set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            if ubiquitous.Light && ubiquitous.VisibleInView then
                set.Add ubiquitous |> ignore<bool>
        Octnode.getLightsInViewBox box set tree.Node

    /// Get all of the elements in a tree that are in a node intersected by one of the given box or frustum depending on its attributes.
    let getElementsInPlay playBox playFrustum (set : _ HashSet) tree =
        for imposter in tree.Imposter do
            set.Add imposter |> ignore<bool>
        for omnipresent in tree.Omnipresent do
            set.Add omnipresent |> ignore<bool>
        for omnipresent in tree.OmnipresentInPlayOnly do
            set.Add omnipresent |> ignore<bool>
        for ubiquitous in tree.UbiquitousFallback do
            set.Add ubiquitous |> ignore<bool>
        Octnode.getElementsInPlay playBox playFrustum set tree.Node

    /// Get the size of the tree's leaves.
    let getLeafSize tree =
        tree.LeafSize

    /// Get the depth of the tree.
    let getDepth tree =
        tree.Depth

    /// Get the bounds of the tree.
    let getBounds tree =
        tree.Bounds

    /// Remove all unused non-root nodes in the tree.
    let sweep tree =
        Octnode.sweep tree.Node

    /// Create an Octree with the given depth and overall size.
    /// Size dimensions must be a power of two.
    let make<'e when 'e : equality> (depth : int) (size : Vector3) =
        if  not (Math.PowerOfTwo size.X) ||
            not (Math.PowerOfTwo size.Y) ||
            not (Math.PowerOfTwo size.Z) then
            failwith "Invalid size for Octree. Expected value whose components are a power of two."
        let leafComparer = // OPTIMIZATION: avoid allocation on Equals calls.
            { new IEqualityComparer<Vector3> with
                member this.Equals (left, right) = left.Equals right
                member this.GetHashCode v = v.GetHashCode () }
        let leaves = dictPlus leafComparer []
        let mutable leafSize = size
        for _ in 0 .. dec depth do leafSize <- leafSize * 0.5f
        let elementComparer = OctelementEqualityComparer<'e> ()
        let min = size * -0.5f + leafSize * 0.5f // OPTIMIZATION: offset min by half leaf size to minimize margin hits at origin.
        let bounds = box3 min size
        { Leaves = leaves
          LeafSize = leafSize
          Imposter = HashSet elementComparer
          Omnipresent = HashSet elementComparer
          OmnipresentInPlayOnly = HashSet elementComparer
          UbiquitousFallback = HashSet elementComparer
          Node = Octnode.make<'e> elementComparer (inc depth) bounds leaves
          Depth = depth
          Bounds = bounds }

/// A spatial structure that organizes elements in a 3d grid.
type Octree<'e when 'e : equality> = Octree.Octree<'e>
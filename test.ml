#!/usr/bin/env owl

#zoo "51eaf74c65fa14c8c466ecfab2351bbd" (* Imagenet_cls *)
#zoo "41bab38b178afd2dd8aa1f0aac2bbc4c" (* ResNet50 CG *)
#require "camlimages.all_formats"

open Owl
module N = Dense.Ndarray.S

(* This file should be run with 'owl ./test.ml' *)

let weight_file = "resnet.weights"
let src = "pics/panda.png"


(* Preprocessing recommended for Resnet. *)
let preprocess img =
  let img = N.copy img in
  let r = N.get_slice [[];[];[];[0]] img in
  let g = N.get_slice [[];[];[];[1]] img in
  let b = N.get_slice [[];[];[];[2]] img in

  N.sub_scalar_ ~out:r r 123.68;
  N.sub_scalar_ ~out:g g 116.779;
  N.sub_scalar_ ~out:b b 103.939;

  N.set_slice [[];[];[];[0]] img b;
  N.set_slice [[];[];[];[1]] img g;
  N.set_slice [[];[];[];[2]] img r;
  img


let convert_to_ndarray src h w =
  let img = Images.load src [] in
  let img = match img with
    | Rgb24 map -> Rgb24.resize None map w h
    | _ -> invalid_arg "conversion not implemented" in
  let res = N.empty [|h; w; 3|] in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let color = Rgb24.get img j i in
      N.set res [|i;j;0|] (float color.r);
      N.set res [|i;j;1|] (float color.g);
      N.set res [|i;j;2|] (float color.b);
    done;
  done;
  res


let prediction src =
  let module R = Resnet in
  let module Graph = R.Compiler.Neural.Graph in
  let module Symbol = R.N.Symbol in
  let module Algodiff = R.Compiler.Neural.Algodiff in
  let pack_arr x = Symbol.pack_arr x |> Algodiff.pack_arr in
  let unpack_arr x = Algodiff.unpack_arr x |> Symbol.unpack_arr in
  let img_size = 299 in
  let nn = R.resnet50 img_size 1000 in
  Graph.load_weights nn weight_file;
  (* Graph.print nn; *)
  let img_arr = convert_to_ndarray src img_size img_size in
  let img_arr = (N.expand img_arr 4) |> preprocess |> pack_arr in
  R.Compiler.model nn img_arr |> unpack_arr


let () =
  Imagenet_cls.to_json (prediction src)
  |> Printf.printf "%s\n"

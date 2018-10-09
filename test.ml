#!/usr/bin/env owl

#zoo "51eaf74c65fa14c8c466ecfab2351bbd" (* Imagenet_cls *)
#require "camlimages.all_formats"

open Owl
open Neural.S
module N = Dense.Ndarray.S

(* This file should be run with 'owl ./test.ml' *)
let weight_file = "weights/resnet.network"
let src = "pics/lion.jpg"

(* Preprocessing recommended for Resnet. *)
let preprocess img =
  let img = N.copy img in
  let r = N.get_slice [[];[];[];[0]] img in
  let g = N.get_slice [[];[];[];[1]] img in
  let b = N.get_slice [[];[];[];[2]] img in

  let r = N.sub_scalar r 123.68 in
  let g = N.sub_scalar g 116.779 in
  let b = N.sub_scalar b 103.939 in

  N.set_slice [[];[];[];[0]] img b;
  N.set_slice [[];[];[];[1]] img g;
  N.set_slice [[];[];[];[2]] img r;
  img

let convert_to_ndarray src h w =
  let img = Images.load src [] in
  let img = match img with
    | Rgb24 map -> Rgb24.resize None map w h
    | _ -> invalid_arg "not implemented conversion" in
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
  let img_size = 224 in
  let nn = Graph.load weight_file in
  (* Graph.print nn; *)
  let img_arr = convert_to_ndarray src img_size img_size in
  let img_arr = (N.expand img_arr 4) |> preprocess in
  Graph.model nn img_arr

let () =
  Imagenet_cls.to_json (prediction src)
  |> Printf.printf "%s\n"

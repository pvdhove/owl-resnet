#!/usr/bin/env owl

#zoo "51eaf74c65fa14c8c466ecfab2351bbd" (* Imagenet_cls *)
#zoo "86a1748bbc898f2e42538839edba00e1" (* ImageUtils *)
#zoo "a05bf0dbe62361b9c2aff89d26d09ba1" (* ResNet50 *)
#require "graphics"
#require "camlimages.graphics"
#require "camlimages.all_formats"

open Owl
open Neural.S
module N = Dense.Ndarray.S

(* This file should be run with 'owl ./test.ml' *)
let weight_file = "weights/resnet.network"
let src = "your/picture.jpg"

(* Preprocessing recommended for Resnet. *)
let preprocess img =
  let img = N.copy img in
  let r = N.get_slice [[];[];[];[0]] img in
  let r = N.sub_scalar r 123.68 in

  let g = N.get_slice [[];[];[];[1]] img in
  let g = N.sub_scalar g 116.779 in

  let b = N.get_slice [[];[];[];[2]] img in
  let b = N.sub_scalar b 103.939 in

  N.set_slice [[];[];[];[0]] img b;
  N.set_slice [[];[];[];[1]] img g;
  N.set_slice [[];[];[];[2]] img r;
  img

let convert_to_ndarray src h w =
  let comp k n = (n lsr ((2 - k) lsl 3)) land 0x0000FF in (* get the kth color component *)
  let img = Images.load src [] in
  let img = match img with
    | Rgb24 map -> Rgb24.resize None map w h
    | _ -> invalid_arg "not implemented yet" in (* TODO *)
  let img_arr = Graphic_image.array_of_image (Rgb24 img) in
  N.init_nd [|h; w; 3|]
    (fun t -> float (comp t.(2) img_arr.(t.(0)).(t.(1))))

let prediction src =
  let img_size = 224 in
  let nn = Graph.load weight_file in
  (* Graph.print nn; *)
  let img_arr = convert_to_ndarray src img_size img_size in
  (* quick hack to replace zero_padding2d *)
  let img_arr = N.pad ~v:0. [[3;3];[3;3];[0;0]] img_arr in
  let img_arr = ImageUtils.(img_arr |> extend_dim) |> preprocess in
  Graph.model nn img_arr

let () =
  Imagenet_cls.to_json (prediction src)
  |> Printf.printf "%s\n"

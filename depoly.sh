#!/bin/bash

yarn global add parcel-bundler

(
cd webcode/ganshowcase
yarn
yarn build
cp -r dist ../../docs
)



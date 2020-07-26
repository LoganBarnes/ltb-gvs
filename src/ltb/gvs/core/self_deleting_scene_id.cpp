// ///////////////////////////////////////////////////////////////////////////////////////
//                                                                           |________|
//  Copyright (c) 2020 CloudNC Ltd - All Rights Reserved                        |  |
//                                                                              |__|
//        ____                                                                .  ||
//       / __ \                                                               .`~||$$$$
//      | /  \ \         /$$$$$$  /$$                           /$$ /$$   /$$  /$$$$$$$
//      \ \ \ \ \       /$$__  $$| $$                          | $$| $$$ | $$ /$$__  $$
//    / / /  \ \ \     | $$  \__/| $$  /$$$$$$  /$$   /$$  /$$$$$$$| $$$$| $$| $$  \__/
//   / / /    \ \__    | $$      | $$ /$$__  $$| $$  | $$ /$$__  $$| $$ $$ $$| $$
//  / / /      \__ \   | $$      | $$| $$  \ $$| $$  | $$| $$  | $$| $$  $$$$| $$
// | | / ________ \ \  | $$    $$| $$| $$  | $$| $$  | $$| $$  | $$| $$\  $$$| $$    $$
//  \ \_/ ________/ /  |  $$$$$$/| $$|  $$$$$$/|  $$$$$$/|  $$$$$$$| $$ \  $$|  $$$$$$/
//   \___/ ________/    \______/ |__/ \______/  \______/  \_______/|__/  \__/ \______/
//
// ///////////////////////////////////////////////////////////////////////////////////////
#include "self_deleting_scene_id.hpp"

namespace ltb {
namespace gvs {

SelfDeletingSceneID::SelfDeletingSceneID(Scene* /*scene*/, SceneId scene_id)
    : scene_id_(std::shared_ptr<SceneId>(new SceneId(scene_id), [/*scene*/](SceneId* scene_id_ptr) {
          // The shared pointer contains the scene removal code so we don't have to implement
          // a reference counter and special move/copy constructors.

          //          if (scene && *scene_id_ptr != nil_id) {
          //              scene->remove_item(*scene_id_ptr);
          //          }

          delete scene_id_ptr;
      })) {}

// The shared pointer contains the `scene->remove_item` functionality
SelfDeletingSceneID::~SelfDeletingSceneID() = default;

auto SelfDeletingSceneID::raw_id() const -> const SceneId& {
    return *scene_id_;
}

} // namespace gvs
} // namespace ltb

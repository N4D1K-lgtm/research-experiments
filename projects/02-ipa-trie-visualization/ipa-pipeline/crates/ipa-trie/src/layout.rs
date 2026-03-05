//! 3D cone-tree layout algorithm using golden angle distribution.

use crate::trie::TrieNode;
use std::f64::consts::PI;

/// Golden angle in radians ≈ 2.39996
// PI * (3 - sqrt(5)) ≈ 2.39996322972865
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_65;

/// Base edge length from parent to child.
const EDGE_BASE: f64 = 14.0;

/// Decay factor per depth level.
const DECAY: f64 = 0.84;

/// Layout engine for 3D cone-tree positioning.
pub struct ConeTreeLayout;

impl ConeTreeLayout {
    /// Apply 3D cone-tree layout to the trie, assigning positions to all nodes.
    pub fn apply(root: &mut TrieNode) {
        root.position = [0.0, 0.0, 0.0];

        if root.children.is_empty() {
            return;
        }

        // Depth 1: distribute on unit sphere using Fibonacci sphere
        let n = root.children.len();
        let children: Vec<String> = root.children.keys().cloned().collect();

        for (i, key) in children.iter().enumerate() {
            let theta = ((1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos()).max(0.0);
            let phi = GOLDEN_ANGLE * i as f64;

            let x = theta.sin() * phi.cos() * EDGE_BASE;
            let y = theta.sin() * phi.sin() * EDGE_BASE;
            let z = theta.cos() * EDGE_BASE;

            let direction = [x, y, z];
            let child = root.children.get_mut(key).unwrap();
            child.position = [x, y, z];

            // Recurse for deeper levels
            let dir_norm = normalize(direction);
            layout_recursive(child, dir_norm, 2);
        }
    }
}

fn layout_recursive(node: &mut TrieNode, parent_direction: [f64; 3], depth: u32) {
    if node.children.is_empty() {
        return;
    }

    let n = node.children.len();
    let edge_length = EDGE_BASE * DECAY.powi((depth - 1) as i32);
    let cone_half_angle = (PI / 2.5).min(PI / 5.0 + 0.04 * n as f64);

    // Build orthonormal frame from parent direction
    let (right, forward) = orthonormal_frame(parent_direction);

    // Sort children by weight for consistent layout
    let mut child_keys: Vec<String> = node.children.keys().cloned().collect();
    child_keys.sort_by(|a, b| {
        let wa = node.children[a].weight;
        let wb = node.children[b].weight;
        wb.cmp(&wa)
    });

    // Distribute children in a cone around the parent direction
    let total_weight: u64 = child_keys
        .iter()
        .map(|k| node.children[k].weight.max(1))
        .sum();

    let mut cumulative_angle = 0.0;

    for key in &child_keys {
        let child = node.children.get(key).unwrap();
        let weight_frac = child.weight.max(1) as f64 / total_weight as f64;
        let angular_wedge = 2.0 * PI * weight_frac;
        let angle = cumulative_angle + angular_wedge / 2.0;
        cumulative_angle += angular_wedge;

        // Compute direction in cone
        let cone_r = cone_half_angle.sin();
        let cone_z = cone_half_angle.cos();

        let dx = cone_r * angle.cos();
        let dy = cone_r * angle.sin();

        // Transform to world coordinates
        let dir = [
            parent_direction[0] * cone_z + right[0] * dx + forward[0] * dy,
            parent_direction[1] * cone_z + right[1] * dx + forward[1] * dy,
            parent_direction[2] * cone_z + right[2] * dx + forward[2] * dy,
        ];

        let dir_n = normalize(dir);
        let pos = [
            node.position[0] + dir_n[0] * edge_length,
            node.position[1] + dir_n[1] * edge_length,
            node.position[2] + dir_n[2] * edge_length,
        ];

        let child_mut = node.children.get_mut(key).unwrap();
        child_mut.position = pos;

        layout_recursive(child_mut, dir_n, depth + 1);
    }
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn orthonormal_frame(direction: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let d = normalize(direction);

    // Pick an arbitrary vector not parallel to d
    let up = if d[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    // right = d × up
    let right = normalize(cross(d, up));
    // forward = d × right
    let forward = normalize(cross(d, right));

    (right, forward)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

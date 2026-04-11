#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use super::convolution::ConvolutionCache;
use super::data::CompatDataPoint;
use super::tree_ids::{DataPointId, NodeId};
use super::tree_node::CompatTreeNode;
use super::tree_stats::{
    CompatNodeCluster, CompatOutlierPoint, CompatTreeLikelihood, CompatTreePriorStats,
};

#[derive(Clone, Debug)]
pub struct CompatTree {
    pub root_node_id: NodeId,
    pub num_samples: usize,
    pub grid_size: usize,
    pub log_prior: f64,
    pub nodes: HashMap<NodeId, CompatTreeNode>,
    parent_by_node: HashMap<NodeId, NodeId>,
    children_by_node: HashMap<NodeId, Vec<NodeId>>,
    cache: ConvolutionCache,
    /// Data point indices that are currently assigned to the outlier node.
    pub assigned_outliers: HashSet<DataPointId>,
}

impl CompatTree {
    pub fn new(
        root_node_id: impl Into<NodeId>,
        num_samples: usize,
        grid_size: usize,
        log_prior: f64,
        cache_entries: usize,
    ) -> Self {
        let root_node_id = root_node_id.into();
        let mut nodes = HashMap::new();
        nodes.insert(
            root_node_id.clone(),
            CompatTreeNode::new(root_node_id.clone(), num_samples, grid_size, log_prior),
        );

        let mut children_by_node = HashMap::new();
        children_by_node.insert(root_node_id.clone(), Vec::new());

        Self {
            root_node_id,
            num_samples,
            grid_size,
            log_prior,
            nodes,
            parent_by_node: HashMap::new(),
            children_by_node,
            cache: ConvolutionCache::new(cache_entries),
            assigned_outliers: HashSet::new(),
        }
    }

    pub fn add_child_node(
        &mut self,
        parent_id: &str,
        child_id: impl Into<NodeId>,
    ) -> Result<(), String> {
        if !self.nodes.contains_key(parent_id) {
            return Err("parent node does not exist".to_string());
        }

        let child_id = child_id.into();
        if self.nodes.contains_key(&child_id) {
            return Err("child node already exists".to_string());
        }

        self.nodes.insert(
            child_id.clone(),
            CompatTreeNode::new(
                child_id.clone(),
                self.num_samples,
                self.grid_size,
                self.log_prior,
            ),
        );
        self.parent_by_node
            .insert(child_id.clone(), parent_id.to_string());
        self.children_by_node
            .entry(parent_id.to_string())
            .or_default()
            .push(child_id.clone());
        self.children_by_node.insert(child_id, Vec::new());

        Ok(())
    }

    pub fn remove_subtree(&mut self, subtree_root_id: &str) -> Result<(), String> {
        if subtree_root_id == self.root_node_id {
            return Err("cannot remove subtree at root".to_string());
        }
        if !self.nodes.contains_key(subtree_root_id) {
            return Err("subtree root node does not exist".to_string());
        }

        let parent_id = self
            .parent_by_node
            .remove(subtree_root_id)
            .ok_or_else(|| "subtree root is already detached".to_string())?;

        if let Some(children) = self.children_by_node.get_mut(&parent_id) {
            children.retain(|id| id != subtree_root_id);
        }

        Ok(())
    }

    pub fn add_subtree(&mut self, parent_id: &str, subtree_root_id: &str) -> Result<(), String> {
        if !self.nodes.contains_key(parent_id) {
            return Err("parent node does not exist".to_string());
        }
        if !self.nodes.contains_key(subtree_root_id) {
            return Err("subtree root node does not exist".to_string());
        }
        if parent_id == subtree_root_id {
            return Err("cannot attach subtree to itself".to_string());
        }
        if self.parent_by_node.contains_key(subtree_root_id) {
            return Err("subtree root must be detached before attach".to_string());
        }

        let descendants = self.descendants_of(subtree_root_id)?;
        if descendants.iter().any(|id| id == parent_id) {
            return Err("cannot create cycle by attaching under descendant".to_string());
        }

        self.parent_by_node
            .insert(subtree_root_id.to_string(), parent_id.to_string());
        self.children_by_node
            .entry(parent_id.to_string())
            .or_default()
            .push(subtree_root_id.to_string());

        Ok(())
    }

    pub fn relabel_node(&mut self, old_id: &str, new_id: impl Into<NodeId>) -> Result<(), String> {
        let new_id = new_id.into();

        if !self.nodes.contains_key(old_id) {
            return Err("node to relabel does not exist".to_string());
        }
        if self.nodes.contains_key(&new_id) {
            return Err("new node id already exists".to_string());
        }

        let mut node = self
            .nodes
            .remove(old_id)
            .ok_or_else(|| "node to relabel does not exist".to_string())?;
        node.node_id = new_id.clone();
        self.nodes.insert(new_id.clone(), node);

        if self.root_node_id == old_id {
            self.root_node_id = new_id.clone();
        }

        if let Some(parent_id) = self.parent_by_node.remove(old_id) {
            self.parent_by_node.insert(new_id.clone(), parent_id);
        }

        for parent in self.parent_by_node.values_mut() {
            if parent == old_id {
                *parent = new_id.clone();
            }
        }

        if let Some(children) = self.children_by_node.remove(old_id) {
            self.children_by_node.insert(new_id.clone(), children);
        }

        for children in self.children_by_node.values_mut() {
            for child_id in children.iter_mut() {
                if child_id == old_id {
                    *child_id = new_id.clone();
                }
            }
        }

        Ok(())
    }

    pub fn parent_of(&self, node_id: &str) -> Result<Option<&str>, String> {
        if !self.nodes.contains_key(node_id) {
            return Err("node does not exist".to_string());
        }
        Ok(self.parent_by_node.get(node_id).map(|id| id.as_str()))
    }

    pub fn children_of(&self, node_id: &str) -> Result<Vec<&str>, String> {
        if !self.nodes.contains_key(node_id) {
            return Err("node does not exist".to_string());
        }

        Ok(self
            .children_by_node
            .get(node_id)
            .map(|ids| ids.iter().map(|id| id.as_str()).collect())
            .unwrap_or_default())
    }

    pub fn descendants_of(&self, node_id: &str) -> Result<Vec<NodeId>, String> {
        if !self.nodes.contains_key(node_id) {
            return Err("node does not exist".to_string());
        }

        let mut stack: Vec<NodeId> = self
            .children_by_node
            .get(node_id)
            .map(|children| children.iter().rev().cloned().collect())
            .unwrap_or_default();
        let mut descendants = Vec::new();

        while let Some(current) = stack.pop() {
            descendants.push(current.clone());
            if let Some(children) = self.children_by_node.get(&current) {
                for child in children.iter().rev() {
                    stack.push(child.clone());
                }
            }
        }

        Ok(descendants)
    }

    pub fn multiplicity(&self, node_id: &str) -> Result<usize, String> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| "node does not exist".to_string())?;
        Ok(node.data_point_ids.len())
    }

    pub fn node(&self, node_id: &str) -> Option<&CompatTreeNode> {
        self.nodes.get(node_id)
    }

    pub fn node_mut(&mut self, node_id: &str) -> Option<&mut CompatTreeNode> {
        self.nodes.get_mut(node_id)
    }

    pub fn update_path_to_root(&mut self, source_id: &str) -> Result<(), String> {
        if !self.nodes.contains_key(source_id) {
            return Err("source node does not exist".to_string());
        }

        let path = self.path_source_to_root(source_id)?;

        for node_id in path {
            self.update_node(&node_id)?;
        }

        Ok(())
    }

    /// Recompute all nodes in post-order (leaves -> root).
    ///
    /// This is the safest option after topology edits such as subtree detach/attach,
    /// because it refreshes all affected branches, including the old parent side.
    pub fn update_all_nodes_postorder(&mut self) -> Result<(), String> {
        if !self.nodes.contains_key(&self.root_node_id) {
            return Err("root node not found".to_string());
        }

        let mut postorder = Vec::with_capacity(self.nodes.len());
        let mut seen = HashSet::new();
        let mut stack: Vec<(NodeId, bool)> = vec![(self.root_node_id.clone(), false)];

        while let Some((node_id, expanded)) = stack.pop() {
            if expanded {
                postorder.push(node_id);
                continue;
            }
            if !seen.insert(node_id.clone()) {
                continue;
            }

            stack.push((node_id.clone(), true));
            let children = self
                .children_by_node
                .get(&node_id)
                .cloned()
                .unwrap_or_default();

            for child_id in children.into_iter().rev() {
                if !self.nodes.contains_key(&child_id) {
                    return Err("child node not found".to_string());
                }
                stack.push((child_id, false));
            }
        }

        if postorder.len() != self.nodes.len() {
            return Err("tree has disconnected nodes from root".to_string());
        }

        for node_id in postorder {
            self.update_node(&node_id)?;
        }

        Ok(())
    }

    fn path_source_to_root(&self, source_id: &str) -> Result<Vec<NodeId>, String> {
        let mut path = Vec::new();
        let mut current = source_id.to_string();

        for _ in 0..=self.nodes.len() {
            path.push(current.clone());
            if current == self.root_node_id {
                return Ok(path);
            }

            let Some(parent) = self.parent_by_node.get(&current) else {
                return Err("node is disconnected from root".to_string());
            };
            current = parent.clone();
        }

        Err("cycle detected in tree parents".to_string())
    }

    fn update_node(&mut self, node_id: &str) -> Result<(), String> {
        let child_ids = self
            .children_by_node
            .get(node_id)
            .cloned()
            .unwrap_or_default();
        // children x samples x grid
        let mut child_log_r_values: Vec<Vec<Vec<f64>>> = Vec::with_capacity(child_ids.len());

        for child_id in child_ids {
            let child = self
                .nodes
                .get(&child_id)
                .ok_or_else(|| "child node not found".to_string())?;
            child_log_r_values.push(child.log_r.clone());
        }

        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| "node not found".to_string())?;
        node.update_node_from_child_r_vals_with_cache(&child_log_r_values, Some(&mut self.cache))
    }

    /// Compute the per-sample tree likelihood from the root node's log-R.
    ///
    /// Returns a `CompatTreeLikelihood` whose `data_log_likelihood[s][g]` is
    /// the root node's log-R value for sample `s` at grid index `g`.
    /// This requires that `update_path_to_root` (or equivalent) has been called
    /// after every data-point assignment change.
    pub fn tree_likelihood(&self) -> Result<CompatTreeLikelihood, String> {
        let root = self
            .nodes
            .get(&self.root_node_id)
            .ok_or_else(|| "root node not found".to_string())?;

        let root_children_count = self
            .children_by_node
            .get(&self.root_node_id)
            .map_or(0, |c| c.len());

        Ok(CompatTreeLikelihood {
            root_children_count,
            data_log_likelihood: root.log_r.clone(),
        })
    }

    /// Extract `CompatTreePriorStats` from the current tree topology.
    ///
    /// - `num_nodes`: total number of nodes excluding the root.
    /// - `multiplicity_log`: log of the product of `k!` for each node's out-degree `k`.
    /// - `root_subtree_node_counts`: for each direct child of root, the count of nodes
    ///   in that subtree (child + all its descendants).
    pub fn prior_stats(&self) -> Result<CompatTreePriorStats, String> {
        // Exclude root from the node count (mirrors PhyClone convention)
        let num_nodes = self.nodes.len().saturating_sub(1);

        // multiplicity = sum_n log(out_degree(n)!)
        let mut multiplicity_log = 0.0_f64;
        for children in self.children_by_node.values() {
            let k = children.len();
            for i in 2..=k {
                multiplicity_log += (i as f64).ln();
            }
        }

        // For each child of root: count nodes in that subtree (child + descendants)
        let root_children = self
            .children_by_node
            .get(&self.root_node_id)
            .cloned()
            .unwrap_or_default();

        let mut root_subtree_node_counts = Vec::with_capacity(root_children.len());
        for child_id in &root_children {
            let desc = self.descendants_of(child_id)?;
            root_subtree_node_counts.push(1 + desc.len());
        }

        Ok(CompatTreePriorStats {
            num_nodes,
            multiplicity_log,
            root_subtree_node_counts,
        })
    }

    /// Build `CompatNodeCluster` entries for every non-root node.
    ///
    /// Each entry records how many data points are currently assigned to the node.
    /// Return node-ids of all non-root nodes in an unspecified order.
    pub fn non_root_node_ids(&self) -> impl Iterator<Item = &str> {
        self.nodes
            .keys()
            .filter(move |id| *id != &self.root_node_id)
            .map(|id| id.as_str())
    }

    /// No explicit outlier node is modelled here; `is_outlier_node` is always `false`.
    pub fn node_clusters(&self) -> Vec<CompatNodeCluster> {
        self.nodes
            .iter()
            .filter(|(id, _)| **id != self.root_node_id)
            .map(|(_, node)| CompatNodeCluster {
                data_point_count: node.data_point_ids.len(),
                is_outlier_node: false,
            })
            .collect()
    }

    /// Return the node id that contains `dp_id`, scanning all non-root nodes and outliers.
    ///
    /// Returns `Some("-1")` when the data point is in `assigned_outliers` (matching the
    /// Python PhyClone sentinel `-1` used for outlier assignments). Returns `None` when
    /// the data point has not yet been added to any node or outlier set.
    pub fn node_id_for_data_point(&self, dp_id: DataPointId) -> Option<&str> {
        // Outlier data points are not graph nodes; return the sentinel "-1".
        if self.assigned_outliers.contains(&dp_id) {
            return Some("-1");
        }
        for (node_id, node) in &self.nodes {
            if node_id == &self.root_node_id {
                continue;
            }
            if node.data_point_ids.contains(&dp_id) {
                return Some(node_id.as_str());
            }
        }
        None
    }

    /// Build a complete `DataPointId -> NodeId` map for every non-root node and outlier.
    ///
    /// Outlier data points are included with the sentinel value `"-1"`, consistent with
    /// `node_id_for_data_point`. Useful for constructing the `constrained_path` in
    /// Conditional SMC.
    pub fn data_point_to_node_map(&self) -> HashMap<DataPointId, &str> {
        let mut map = HashMap::new();
        // Include outlier data points with sentinel "-1".
        for &dp_id in &self.assigned_outliers {
            map.insert(dp_id, "-1");
        }
        for (node_id, node) in &self.nodes {
            if node_id == &self.root_node_id {
                continue;
            }
            for &dp_id in &node.data_point_ids {
                map.insert(dp_id, node_id.as_str());
            }
        }
        map
    }

    /// Returns a reference to the set of data point indices currently assigned as outliers.
    pub fn assigned_outlier_ids(&self) -> &HashSet<DataPointId> {
        &self.assigned_outliers
    }

    /// Mark a data point as an outlier.
    pub fn assign_outlier(&mut self, id: DataPointId) {
        self.assigned_outliers.insert(id);
    }

    /// Unmark a data point as an outlier.
    pub fn unassign_outlier(&mut self, id: DataPointId) {
        self.assigned_outliers.remove(&id);
    }

    /// Build `CompatOutlierPoint` entries from a slice of `CompatDataPoint`.
    ///
    /// The returned vec is indexed by `DataPointId` (i.e. `dp.idx`), suitable for
    /// passing to `CompatTreeJointDistribution::log_p` / `log_p_one`.
    pub fn outlier_points(data_points: &[CompatDataPoint]) -> Vec<CompatOutlierPoint> {
        data_points
            .iter()
            .map(|dp| CompatOutlierPoint {
                id: dp.idx,
                outlier_prob: dp.outlier_prob,
                outlier_prob_not: dp.outlier_prob_not,
                outlier_marginal_prob: dp.outlier_marginal_prob,
            })
            .collect()
    }

    // ── Subtree Particle Gibbs helpers ────────────────────────────────────────

    /// Collect all node IDs in the subtree rooted at `subtree_root_id`
    /// (inclusive of the root itself).
    pub fn subtree_node_ids(&self, subtree_root_id: &str) -> Result<Vec<NodeId>, String> {
        if !self.nodes.contains_key(subtree_root_id) {
            return Err(format!("subtree root node {} not found", subtree_root_id));
        }
        let mut ids = vec![subtree_root_id.to_string()];
        ids.extend(self.descendants_of(subtree_root_id)?);
        Ok(ids)
    }

    /// Return all data point IDs currently assigned in this tree (nodes + outliers).
    pub fn all_assigned_data_point_ids(&self) -> Vec<DataPointId> {
        let mut ids: Vec<DataPointId> = self.assigned_outliers.iter().copied().collect();
        for (node_id, node) in &self.nodes {
            if *node_id == self.root_node_id {
                continue;
            }
            ids.extend_from_slice(&node.data_point_ids);
        }
        ids
    }

    /// Extract the subtree rooted at `subtree_root_id` as a standalone `CompatTree`.
    ///
    /// The extracted tree has a fresh dummy root ("root").  `subtree_root_id` and all
    /// its descendants are cloned into the new tree; `subtree_root_id` becomes a direct
    /// child of the new dummy root.  Outliers are NOT moved here – the caller is
    /// responsible for transferring them after extraction.
    pub fn extract_subtree_tree(&self, subtree_root_id: &str) -> Result<CompatTree, String> {
        if !self.nodes.contains_key(subtree_root_id) {
            return Err(format!("subtree root node {} not found", subtree_root_id));
        }

        let ids = self.subtree_node_ids(subtree_root_id)?;

        // Guard: node named "root" would clash with the new dummy root.
        for id in &ids {
            if id.as_str() == "root" {
                return Err(
                    "subtree contains a node called 'root'; cannot extract (id conflict)"
                        .to_string(),
                );
            }
        }

        let cache_entries = (ids.len() + 2).next_power_of_two().max(8);
        let mut new_tree = CompatTree::new(
            "root",
            self.num_samples,
            self.grid_size,
            self.log_prior,
            cache_entries,
        );

        let id_set: HashSet<NodeId> = ids.iter().cloned().collect();

        // Insert all subtree nodes (cloned).
        for id in &ids {
            let node = self
                .nodes
                .get(id)
                .ok_or_else(|| format!("node {} missing during extraction", id))?
                .clone();
            new_tree.nodes.insert(id.clone(), node);
            // Copy only edges that stay within the extracted subtree.
            let children = self
                .children_by_node
                .get(id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|child| id_set.contains(child))
                .collect();
            new_tree.children_by_node.insert(id.clone(), children);
        }

        // Set parent relationships.
        // subtree_root_id → new dummy root.
        new_tree
            .parent_by_node
            .insert(subtree_root_id.to_string(), "root".to_string());
        new_tree
            .children_by_node
            .get_mut("root")
            .unwrap()
            .push(subtree_root_id.to_string());

        // All other subtree nodes keep their original parent.
        for id in &ids {
            if id.as_str() == subtree_root_id {
                continue;
            }
            if let Some(parent) = self.parent_by_node.get(id) {
                new_tree.parent_by_node.insert(id.clone(), parent.clone());
            }
        }

        new_tree.update_all_nodes_postorder()?;
        Ok(new_tree)
    }

    /// Remove the subtree rooted at `subtree_root_id` from this tree.
    ///
    /// Returns `Ok(Some(parent_id))` where `parent_id` is the node that `subtree_root_id`
    /// was attached to, or `Ok(None)` if `subtree_root_id` was a direct child of the
    /// tree root (parent = root node id itself).
    ///
    /// Unlike `remove_subtree` (which only detaches), this method also physically
    /// removes all subtree nodes from `self.nodes`, `parent_by_node`, and
    /// `children_by_node`.
    pub fn remove_subtree_nodes(
        &mut self,
        subtree_root_id: &str,
    ) -> Result<Option<NodeId>, String> {
        if subtree_root_id == self.root_node_id {
            return Err("cannot remove the tree root via remove_subtree_nodes".to_string());
        }
        if !self.nodes.contains_key(subtree_root_id) {
            return Err(format!("subtree root node {} not found", subtree_root_id));
        }

        // Remember where the subtree root was attached.
        let parent_id = self.parent_by_node.get(subtree_root_id).cloned();

        // Collect all nodes to delete.
        let ids = self.subtree_node_ids(subtree_root_id)?;

        // Disconnect the subtree root from its parent's children list.
        if let Some(ref pid) = parent_id {
            if let Some(children) = self.children_by_node.get_mut(pid) {
                children.retain(|id| id != subtree_root_id);
            }
        }

        // Remove all subtree nodes.
        for id in &ids {
            self.nodes.remove(id);
            self.parent_by_node.remove(id);
            self.children_by_node.remove(id);
        }

        // Use a full recomputation after subtree removal. This is more conservative
        // than updating only the former parent's path and keeps the first subtree-PG
        // implementation biased toward correctness over speed.
        self.update_all_nodes_postorder()?;

        // Return the non-root parent as Some(id), or None if the subtree root was
        // attached to the tree root.
        Ok(parent_id.filter(|pid| pid != &self.root_node_id))
    }

    /// Graft `subtree` (a standalone `CompatTree` with a dummy root) into this tree
    /// under `parent_id`.
    ///
    /// All direct children of `subtree`'s dummy root become children of `parent_id`
    /// (or of `self.root_node_id` when `parent_id` is `None`).
    ///
    /// Node ID collisions between the subtree and the current tree are resolved by
    /// renaming the colliding subtree nodes to fresh names ("graft-1", "graft-2", …).
    ///
    /// After grafting, `update_all_nodes_postorder` is called to refresh all cached
    /// likelihood values.
    pub fn graft_subtree_tree(
        &mut self,
        subtree: &CompatTree,
        parent_id: Option<&str>,
    ) -> Result<(), String> {
        let subtree_root_children: Vec<NodeId> = subtree
            .children_by_node
            .get(&subtree.root_node_id)
            .cloned()
            .unwrap_or_default();

        if subtree_root_children.is_empty() {
            // Nothing to graft.
            return Ok(());
        }

        // Collect all non-root node IDs from the subtree.
        let mut subtree_node_ids: Vec<NodeId> = Vec::new();
        for child in &subtree_root_children {
            subtree_node_ids.push(child.clone());
            subtree_node_ids.extend(subtree.descendants_of(child)?);
        }

        // Build a rename map for any ID that collides with an existing node.
        let existing_ids: HashSet<NodeId> = self.nodes.keys().cloned().collect();
        let subtree_ids: HashSet<NodeId> = subtree_node_ids.iter().cloned().collect();
        let mut rename_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut counter: usize = 0;

        for node_id in &subtree_node_ids {
            if existing_ids.contains(node_id) || node_id.as_str() == "root" {
                let fresh_id = loop {
                    counter += 1;
                    let candidate = format!("graft-{}", counter);
                    if !existing_ids.contains(&candidate)
                        && !subtree_ids.contains(&candidate)
                        && !rename_map.values().any(|v| v == &candidate)
                    {
                        break candidate;
                    }
                };
                rename_map.insert(node_id.clone(), fresh_id);
            }
        }

        let resolve = |id: &str| -> NodeId {
            rename_map
                .get(id)
                .cloned()
                .unwrap_or_else(|| id.to_string())
        };

        // Insert all subtree nodes into self.
        for node_id in &subtree_node_ids {
            let resolved = resolve(node_id);
            let mut node = subtree
                .nodes
                .get(node_id)
                .ok_or_else(|| format!("subtree node {} not found during graft", node_id))?
                .clone();
            node.node_id = resolved.clone();
            self.nodes.insert(resolved.clone(), node);
            self.children_by_node.insert(resolved, Vec::new());
        }

        // Restore children relationships within the subtree.
        for node_id in &subtree_node_ids {
            let resolved = resolve(node_id);
            let children = subtree
                .children_by_node
                .get(node_id)
                .cloned()
                .unwrap_or_default();
            let resolved_children: Vec<NodeId> = children.iter().map(|c| resolve(c)).collect();
            if let Some(child_list) = self.children_by_node.get_mut(&resolved) {
                *child_list = resolved_children.clone();
            }
            for child_resolved in &resolved_children {
                self.parent_by_node
                    .insert(child_resolved.clone(), resolved.clone());
            }
        }

        // Connect subtree root children under the graft parent.
        let graft_parent = parent_id.unwrap_or(self.root_node_id.as_str()).to_string();

        if !self.nodes.contains_key(&graft_parent) {
            return Err(format!("graft parent {} not found", graft_parent));
        }

        for child_id in &subtree_root_children {
            let resolved_child = resolve(child_id);
            self.parent_by_node
                .insert(resolved_child.clone(), graft_parent.clone());
            self.children_by_node
                .entry(graft_parent.clone())
                .or_default()
                .push(resolved_child);
        }

        // Copy outliers from the subtree.
        for &dp_id in &subtree.assigned_outliers {
            self.assigned_outliers.insert(dp_id);
        }

        self.update_all_nodes_postorder()?;
        Ok(())
    }
}

/// Builds a star-topology [`CompatTree`] from a slice of `(node_id, data_point_value)` pairs.
///
/// All cluster nodes are direct children of root. Each node receives the corresponding
/// DataPoint likelihood tensor (`value: &[Vec<f64>]`, samples x grid). The root node
/// holds no DataPoints.
///
/// `log_prior` is the per-node log-prior (typically `0.0` or a uniform value).
pub fn build_star_tree(
    data_points: &[(usize, &str, &[Vec<f64>])],
    num_samples: usize,
    grid_size: usize,
    log_prior: f64,
) -> Result<CompatTree, String> {
    let cache_entries = (data_points.len() + 1).next_power_of_two().max(8);
    let mut tree = CompatTree::new("root", num_samples, grid_size, log_prior, cache_entries);

    for (idx, node_id, value) in data_points {
        tree.add_child_node("root", *node_id)?;
        tree.node_mut(node_id)
            .ok_or_else(|| format!("node {} missing after add", node_id))?
            .add_data_point(*idx, value)?;
        tree.update_path_to_root(node_id)?;
    }

    Ok(tree)
}

/// Builds a PhyClone-style single-node initial tree.
///
/// All data points are assigned to one ordinary node under the dummy root.
pub fn build_single_node_tree(
    data_points: &[CompatDataPoint],
    num_samples: usize,
    grid_size: usize,
    log_prior: f64,
) -> Result<CompatTree, String> {
    if data_points.is_empty() {
        return Err("cannot build single-node tree from empty data".to_string());
    }

    let cache_entries = (data_points.len() + 1).next_power_of_two().max(8);
    let mut tree = CompatTree::new("root", num_samples, grid_size, log_prior, cache_entries);

    // Keep this aligned with `CompatTreeShellNodeAdder` allocation (`shell-{idx}`),
    // so conditional SMC constrained paths match proposal node ids.
    let node_id = "shell-1";
    tree.add_child_node("root", node_id)?;

    for dp in data_points {
        tree.node_mut(node_id)
            .ok_or_else(|| "single-node tree node disappeared".to_string())?
            .add_data_point(dp.idx, &dp.value)?;
    }

    tree.update_path_to_root(node_id)?;
    Ok(tree)
}

/// Builds a partial-chain [`CompatTree`] from a slice of `(node_id, data_point_value)` pairs.
///
/// The first two cluster nodes are linked as root -> child0 -> child1. All remaining nodes
/// are direct children of root (star). This gives two distinct topologies when alternated
/// with `build_star_tree`.
pub fn build_partial_chain_tree(
    data_points: &[(usize, &str, &[Vec<f64>])],
    num_samples: usize,
    grid_size: usize,
    log_prior: f64,
) -> Result<CompatTree, String> {
    if data_points.len() < 2 {
        return build_star_tree(data_points, num_samples, grid_size, log_prior);
    }

    let cache_entries = (data_points.len() + 1).next_power_of_two().max(8);
    let mut tree = CompatTree::new("root", num_samples, grid_size, log_prior, cache_entries);

    // First node: child of root
    let (first_idx, first_id, first_val) = data_points[0];
    tree.add_child_node("root", first_id)?;
    tree.node_mut(first_id)
        .ok_or_else(|| format!("node {} missing after add", first_id))?
        .add_data_point(first_idx, first_val)?;
    tree.update_path_to_root(first_id)?;

    // Second node: child of first
    let (second_idx, second_id, second_val) = data_points[1];
    tree.add_child_node(first_id, second_id)?;
    tree.node_mut(second_id)
        .ok_or_else(|| format!("node {} missing after add", second_id))?
        .add_data_point(second_idx, second_val)?;
    tree.update_path_to_root(second_id)?;

    // Remaining nodes: direct children of root
    for (idx, node_id, value) in data_points.iter().skip(2) {
        tree.add_child_node("root", *node_id)?;
        tree.node_mut(node_id)
            .ok_or_else(|| format!("node {} missing after add", node_id))?
            .add_data_point(*idx, value)?;
        tree.update_path_to_root(node_id)?;
    }

    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::{build_star_tree, CompatTree};
    use crate::phyclone::compat::convolution::compute_log_s_1d;
    use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

    fn lse(values: &[f64]) -> f64 {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            return max;
        }
        let sum: f64 = values.iter().map(|v| (*v - max).exp()).sum();
        max + sum.ln()
    }

    fn assert_all_data_points_assigned_once(tree: &CompatTree, data_points: &[CompatDataPoint]) {
        let mut seen = std::collections::HashSet::new();

        for node_id in tree.non_root_node_ids() {
            if node_id == "-1" {
                continue;
            }

            if let Some(node) = tree.node(node_id) {
                for dp_id in &node.data_point_ids {
                    assert!(seen.insert(*dp_id), "duplicate dp_id {}", dp_id);
                }
            }
        }

        for dp_id in &tree.assigned_outliers {
            assert!(seen.insert(*dp_id), "duplicate outlier dp_id {}", dp_id);
        }

        for dp in data_points {
            assert!(seen.contains(&dp.idx), "missing dp_id {}", dp.idx);
        }

        assert_eq!(seen.len(), data_points.len());
    }

    #[test]
    fn update_path_to_root_updates_source_to_root_only() {
        // 1 sample, 3 grid points
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 16);
        tree.add_child_node("root", "a")
            .expect("add a should succeed");
        tree.add_child_node("a", "leaf")
            .expect("add leaf should succeed");
        tree.add_child_node("root", "b")
            .expect("add b should succeed");

        tree.node_mut("leaf").expect("leaf exists").log_p = vec![vec![-0.2, -1.0, -2.0]];
        tree.node_mut("a").expect("a exists").log_p = vec![vec![-0.5, -0.4, -1.0]];
        tree.node_mut("root").expect("root exists").log_p = vec![vec![-0.1, -0.2, -0.3]];
        tree.node_mut("b").expect("b exists").log_r = vec![vec![-0.9, -0.6, -0.4]];

        tree.update_path_to_root("leaf")
            .expect("path update should succeed");

        let leaf_log_r = tree.node("leaf").expect("leaf exists").log_r.clone();
        assert_eq!(leaf_log_r, vec![vec![-0.2, -1.0, -2.0]]);

        let log_d0 = leaf_log_r[0][0];
        let log_d1 = leaf_log_r[0][1];
        let log_d2 = leaf_log_r[0][2];
        let a_expected_s0 = vec![
            -0.5 + log_d0,
            -0.4 + lse(&[log_d0, log_d1]),
            -1.0 + lse(&[log_d0, log_d1, log_d2]),
        ];
        let a_actual = tree.node("a").expect("a exists").log_r.clone();
        for (actual, expected) in a_actual[0].iter().zip(a_expected_s0.iter()) {
            assert!((actual - expected).abs() <= 1e-10);
        }

        let b_log_r = tree.node("b").expect("b exists").log_r.clone();
        assert_eq!(b_log_r, vec![vec![-0.9, -0.6, -0.4]]);

        let root_log_s = compute_log_s_1d(&[a_actual[0].clone(), b_log_r[0].clone()])
            .expect("root log_s should succeed");
        let root_expected_s0 = vec![
            -0.1 + root_log_s[0],
            -0.2 + root_log_s[1],
            -0.3 + root_log_s[2],
        ];
        let root_actual = tree.node("root").expect("root exists").log_r.clone();
        for (actual, expected) in root_actual[0].iter().zip(root_expected_s0.iter()) {
            assert!((actual - expected).abs() <= 1e-10);
        }
    }

    #[test]
    fn update_path_to_root_on_root_updates_only_root() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "c1")
            .expect("add c1 should succeed");

        tree.node_mut("root").expect("root exists").log_p = vec![vec![-0.2, -0.2, -0.2]];
        tree.node_mut("c1").expect("c1 exists").log_r = vec![vec![-1.0, -0.5, -0.1]];

        tree.update_path_to_root("root")
            .expect("root path update should succeed");

        let log_s = compute_log_s_1d(&[vec![-1.0, -0.5, -0.1]]).expect("log_s should succeed");
        let expected_s0 = vec![-0.2 + log_s[0], -0.2 + log_s[1], -0.2 + log_s[2]];
        let actual = tree.node("root").expect("root exists").log_r.clone();

        for (a, e) in actual[0].iter().zip(expected_s0.iter()) {
            assert!((a - e).abs() <= 1e-10);
        }
    }

    #[test]
    fn update_path_to_root_errors_on_unknown_source() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        let err = tree
            .update_path_to_root("unknown")
            .expect_err("unknown source must error");
        assert!(err.contains("source node"));
    }

    #[test]
    fn update_all_nodes_postorder_recomputes_root() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "a")
            .expect("add a should succeed");
        tree.add_child_node("a", "leaf")
            .expect("add leaf should succeed");

        let value = vec![vec![-1.0, -0.7, -0.2]];
        tree.node_mut("leaf")
            .expect("leaf exists")
            .add_data_point(0, &value)
            .expect("add datapoint should succeed");

        tree.update_all_nodes_postorder()
            .expect("full update should succeed");

        let root = tree.node("root").expect("root exists");
        assert!(root.log_r.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn remove_and_add_subtree_detaches_and_reattaches_branch() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "a")
            .expect("add a should succeed");
        tree.add_child_node("a", "leaf")
            .expect("add leaf should succeed");

        tree.remove_subtree("a")
            .expect("remove subtree should succeed");
        assert_eq!(tree.parent_of("a").expect("a must exist"), None);
        let root_children = tree.children_of("root").expect("root must exist");
        assert!(root_children.is_empty());

        tree.add_subtree("root", "a")
            .expect("reattach subtree should succeed");
        assert_eq!(tree.parent_of("a").expect("a must exist"), Some("root"));
        let root_children = tree.children_of("root").expect("root must exist");
        assert_eq!(root_children, vec!["a"]);
    }

    #[test]
    fn extract_remove_graft_roundtrip_preserves_assignments() {
        let data_points: Vec<CompatDataPoint> = (0..3)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0 - 0.1 * i as f64, -0.5 - 0.1 * i as f64, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let log_prior = -(data_points[0].value[0].len() as f64).ln();
        let mut tree = CompatTree::new("root", 1, data_points[0].value[0].len(), log_prior, 8);
        tree.add_child_node("root", "A").expect("add A");
        tree.add_child_node("A", "B").expect("add B");
        tree.add_child_node("root", "C").expect("add C");

        tree.node_mut("A")
            .expect("A exists")
            .add_data_point(data_points[0].idx, &data_points[0].value)
            .expect("assign dp0 to A");
        tree.node_mut("B")
            .expect("B exists")
            .add_data_point(data_points[1].idx, &data_points[1].value)
            .expect("assign dp1 to B");
        tree.node_mut("C")
            .expect("C exists")
            .add_data_point(data_points[2].idx, &data_points[2].value)
            .expect("assign dp2 to C");
        tree.update_all_nodes_postorder()
            .expect("initial tree update");

        let subtree = tree.extract_subtree_tree("A").expect("extract subtree A");
        assert_all_data_points_assigned_once(&subtree, &data_points[..2]);

        let mut pruned = tree.clone();
        let parent = pruned.remove_subtree_nodes("A").expect("remove subtree A");
        assert_eq!(parent, None);
        assert_all_data_points_assigned_once(&pruned, &data_points[2..]);

        pruned
            .graft_subtree_tree(&subtree, parent.as_deref())
            .expect("graft subtree A back");
        assert_all_data_points_assigned_once(&pruned, &data_points);
    }

    #[test]
    fn graft_subtree_tree_avoids_fresh_id_collision_with_existing_subtree_ids() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "A")
            .expect("add A to base tree");

        let mut subtree = CompatTree::new("subroot", 1, 3, -1.0, 8);
        subtree
            .add_child_node("subroot", "A")
            .expect("add colliding A to subtree");
        subtree
            .add_child_node("subroot", "graft-1")
            .expect("add existing graft-1 to subtree");

        tree.graft_subtree_tree(&subtree, None)
            .expect("graft should succeed without id collisions");

        assert!(tree.nodes.contains_key("A"));
        assert!(tree.nodes.contains_key("graft-1"));
        assert!(tree.nodes.contains_key("graft-2"));

        let root_children = tree.children_of("root").expect("root must exist");
        assert_eq!(root_children.len(), 3);
        assert!(root_children.contains(&"A"));
        assert!(root_children.contains(&"graft-1"));
        assert!(root_children.contains(&"graft-2"));
    }

    #[test]
    fn descendants_and_multiplicity_work() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "a")
            .expect("add a should succeed");
        tree.add_child_node("root", "b")
            .expect("add b should succeed");
        tree.add_child_node("a", "leaf")
            .expect("add leaf should succeed");

        tree.node_mut("a")
            .expect("a must exist")
            .add_data_point(7, &[vec![-0.1, -0.2, -0.3]])
            .expect("add datapoint should succeed");

        let descendants = tree.descendants_of("root").expect("root must exist");
        assert_eq!(
            descendants,
            vec!["a".to_string(), "leaf".to_string(), "b".to_string()]
        );
        assert_eq!(tree.multiplicity("a").expect("a must exist"), 1);
        assert_eq!(tree.multiplicity("b").expect("b must exist"), 0);
    }

    #[test]
    fn relabel_node_updates_parent_and_children_links() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "a")
            .expect("add a should succeed");
        tree.add_child_node("a", "leaf")
            .expect("add leaf should succeed");

        tree.relabel_node("a", "a2")
            .expect("relabel should succeed");

        assert!(tree.node("a").is_none());
        assert!(tree.node("a2").is_some());
        assert_eq!(tree.parent_of("a2").expect("a2 must exist"), Some("root"));
        assert_eq!(tree.parent_of("leaf").expect("leaf must exist"), Some("a2"));
        let root_children = tree.children_of("root").expect("root must exist");
        assert_eq!(root_children, vec!["a2"]);
    }

    // -----------------------------------------------------------------------
    // Phase D parity tests: verify log_prior initialization and outlier
    // behaviour against the Python PhyClone design.
    // -----------------------------------------------------------------------

    fn approx_eq(lhs: f64, rhs: f64) {
        let delta = (lhs - rhs).abs();
        assert!(delta <= 1e-10, "lhs={} rhs={} delta={}", lhs, rhs, delta);
    }

    /// Python PhyClone initialises TreeNode.log_p with -np.log(grid_size[1]).
    /// Verify that CompatTree::new with log_prior = -(grid_size as f64).ln()
    /// sets the root node's log_p values to that exact uniform prior.
    #[test]
    fn log_prior_matches_python_uniform_prior() {
        let grid_size = 5;
        let log_prior = -(grid_size as f64).ln();
        let tree = CompatTree::new("root", 1, grid_size, log_prior, 8);
        let root = tree.node("root").expect("root must exist");
        let expected = -(5_f64).ln();
        for g in 0..grid_size {
            approx_eq(root.log_p[0][g], expected);
        }
    }

    /// When a data point is added to a leaf node, log_p should equal
    /// log_prior + data_value, matching Python's `log_p += dp.value`.
    #[test]
    fn node_log_p_accumulates_data_correctly() {
        let grid_size = 3;
        let log_prior = -(grid_size as f64).ln();
        let mut tree = CompatTree::new("root", 1, grid_size, log_prior, 8);
        tree.add_child_node("root", "n0").expect("add n0");

        let data_value = vec![vec![0.5_f64, 0.3, 0.1]];
        tree.node_mut("n0")
            .expect("n0")
            .add_data_point(0, &data_value)
            .expect("add dp");

        let n0 = tree.node("n0").expect("n0");
        approx_eq(n0.log_p[0][0], log_prior + 0.5);
        approx_eq(n0.log_p[0][1], log_prior + 0.3);
        approx_eq(n0.log_p[0][2], log_prior + 0.1);
    }

    /// Outlier data points must be tracked in assigned_outliers only,
    /// NOT added as a graph node. The tree_likelihood must not be affected
    /// by an outlier assignment (Python: outliers are outside the graph).
    #[test]
    fn outlier_assignment_does_not_affect_tree_likelihood() {
        let grid_size = 3;
        let log_prior = -(grid_size as f64).ln();
        let mut tree = CompatTree::new("root", 1, grid_size, log_prior, 8);
        tree.add_child_node("root", "n0").expect("add n0");

        let data_value = vec![vec![0.5_f64, 0.3, 0.1]];
        tree.node_mut("n0")
            .expect("n0")
            .add_data_point(0, &data_value)
            .expect("add dp0");
        tree.update_path_to_root("n0").expect("update path");

        let likelihood_before = tree.tree_likelihood().expect("likelihood");

        // Mark dp1 as an outlier (using assigned_outliers, not a graph node).
        tree.assign_outlier(1);

        let likelihood_after = tree.tree_likelihood().expect("likelihood");

        // Tree likelihood must be unchanged: outliers are outside the graph.
        assert_eq!(
            likelihood_before.data_log_likelihood,
            likelihood_after.data_log_likelihood
        );
        // Outlier node "-1" must NOT exist in the graph.
        assert!(tree.node("-1").is_none());
        // assigned_outliers must contain dp1.
        assert!(tree.assigned_outliers.contains(&1));
    }

    /// node_id_for_data_point returns "-1" for outlier data points and the
    /// actual node id for graph-assigned data points.
    #[test]
    fn node_id_for_data_point_returns_sentinel_for_outliers() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").expect("add n0");
        tree.node_mut("n0")
            .expect("n0")
            .add_data_point(0, &[vec![0.0; 3]])
            .expect("add dp0");
        tree.assign_outlier(99);

        assert_eq!(tree.node_id_for_data_point(0), Some("n0"));
        assert_eq!(tree.node_id_for_data_point(99), Some("-1"));
        assert_eq!(tree.node_id_for_data_point(42), None);
    }

    /// data_point_to_node_map includes both graph-assigned and outlier data points.
    #[test]
    fn data_point_to_node_map_includes_outliers() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").expect("add n0");
        tree.node_mut("n0")
            .expect("n0")
            .add_data_point(0, &[vec![0.0; 3]])
            .expect("add dp0");
        tree.assign_outlier(5);

        let map = tree.data_point_to_node_map();
        assert_eq!(map.get(&0), Some(&"n0"));
        assert_eq!(map.get(&5), Some(&"-1"));
        assert_eq!(map.get(&99), None);
    }

    /// log_p_one (fixed-CCF variant) must be <= log_p (marginalised).
    /// This is a sanity check: fixing a single grid point can only
    /// decrease or maintain the marginalised probability.
    #[test]
    fn log_p_one_is_at_most_log_p_for_star_tree() {
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;

        let grid_size = 5;
        let log_prior = -(grid_size as f64).ln();
        let data_value = vec![vec![-0.8, -0.5, -0.3, -0.6, -1.2]]; // 1 sample x 5 grid

        let tree =
            build_star_tree(&[(0, "n0", &data_value)], 1, grid_size, log_prior).expect("star tree");

        let outlier_points = CompatTree::outlier_points(&[]);
        let joint = CompatTreeJointDistribution::default();

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one_tree(&tree, &outlier_points)
            .expect("compute log_p");

        assert!(
            log_p_one <= log_p + 1e-10,
            "log_p_one={} must be <= log_p={}",
            log_p_one,
            log_p
        );
        assert!(log_p.is_finite(), "log_p must be finite");
        assert!(log_p_one.is_finite(), "log_p_one must be finite");
    }

    /// Adding more data points to a tree increases the number of nodes (for
    /// a star topology with separate nodes per DP) but log_p stays finite.
    #[test]
    fn star_tree_with_multiple_data_points_has_finite_log_p() {
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;

        let grid_size = 4;
        let log_prior = -(grid_size as f64).ln();
        let v0 = vec![vec![-0.3, -0.7, -0.5, -0.9]];
        let v1 = vec![vec![-0.6, -0.4, -0.8, -0.2]];
        let v2 = vec![vec![-0.5, -0.5, -0.5, -0.5]];

        let tree = build_star_tree(
            &[(0, "n0", &v0), (1, "n1", &v1), (2, "n2", &v2)],
            1,
            grid_size,
            log_prior,
        )
        .expect("star tree");

        let outlier_points = CompatTree::outlier_points(&[]);
        let joint = CompatTreeJointDistribution::default();
        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one_tree(&tree, &outlier_points)
            .expect("compute log_p");

        assert!(log_p.is_finite());
        assert!(log_p_one.is_finite());
        // 3-node star: prior_stats.num_nodes = 3
        let stats = tree.prior_stats().expect("prior stats");
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.root_subtree_node_counts.len(), 3);
    }
}

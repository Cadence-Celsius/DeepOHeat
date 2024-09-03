def iterate_over_entire_geometry(geometry, fun, *args):
    geometry = geometry.to_root()

    all_leaf = geometry.is_leaf()
    fun(geometry, *args)

    while not all_leaf:
        children_all_leaf = True
        stem_child_boundary_name, stem_child_idx = None, None

        for boundary_name, children_list in geometry.children.items():
            for child in children_list:
                fun(child, *args)

                if not child.is_leaf() and children_all_leaf:
                    stem_child_boundary_name, stem_child_idx = (
                        boundary_name,
                        geometry.children[boundary_name].index(child),
                    )

                children_all_leaf &= child.is_leaf()

        if not children_all_leaf:
            geometry = geometry.to_child(stem_child_boundary_name, stem_child_idx)
        else:
            last_sibling, next_sibling_boundary, next_sibling_idx = (
                geometry.if_last_sibling()
            )

            while last_sibling and geometry.parent is not None:
                geometry = geometry.to_parent()
                last_sibling, next_sibling_boundary, next_sibling_idx = (
                    geometry.if_last_sibling()
                )

            if last_sibling:
                all_leaf = True
            else:
                geometry = geometry.to_next_sibling(
                    next_sibling_boundary, next_sibling_idx
                )

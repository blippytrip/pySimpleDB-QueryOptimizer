"""
solution.py — Query Optimization & Indexing for pySimpleDB

Implements:
  - BTreeIndex          : in-memory B-Tree supporting insert / search
  - CompositeIndex      : B-Tree keyed on tuples of field values
  - IndexScan           : scan driven by an index instead of full table scan
  - BetterQueryPlanner  : selection pushdown + heuristic join reordering
  - IndexQueryPlanner   : replaces table scans with index scans where possible
  - create_indexes      : builds and populates all indexes from live table data
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from Planner import TablePlan, SelectPlan, ProjectPlan, ProductPlan
from RelationalOp import (
    Predicate, Term, Expression, Constant,
    SelectScan, ProjectScan, ProductScan,
)
from Record import Schema, Layout, TableScan, RecordID
from Metadata import MetadataMgr
from Transaction import Transaction


# ---------------------------------------------------------------------------
# B-Tree implementation (in-memory, supports any comparable key)
# ---------------------------------------------------------------------------

class _BTreeNode:
    """Internal node of the B-Tree."""

    def __init__(self, order: int, leaf: bool = False):
        self.order = order          # max children = order; max keys = order - 1
        self.leaf = leaf
        self.keys: List[Any] = []
        self.values: List[List[Any]] = []   # only used in leaf nodes (list of record-ids per key)
        self.children: List["_BTreeNode"] = []

    # ------------------------------------------------------------------
    # Leaf helpers
    # ------------------------------------------------------------------
    def _leaf_insert(self, key, rid):
        """Insert (key, rid) into this leaf node."""
        for i, k in enumerate(self.keys):
            if key == k:
                self.values[i].append(rid)
                return
            if key < k:
                self.keys.insert(i, key)
                self.values.insert(i, [rid])
                return
        self.keys.append(key)
        self.values.append([rid])

    def _leaf_search(self, key) -> List[Any]:
        for i, k in enumerate(self.keys):
            if k == key:
                return list(self.values[i])
            if k > key:
                return []
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _internal_insert(self, key, left_child, right_child):
        """Insert a promoted key and its right child into this internal node."""
        for i, k in enumerate(self.keys):
            if key < k:
                self.keys.insert(i, key)
                self.children.insert(i + 1, right_child)
                return
        self.keys.append(key)
        self.children.append(right_child)

    def is_full(self) -> bool:
        return len(self.keys) >= self.order - 1

    def split_leaf(self) -> Tuple["_BTreeNode", Any]:
        """Split a full leaf node; return (new_right_node, promoted_key)."""
        mid = len(self.keys) // 2
        right = _BTreeNode(self.order, leaf=True)
        right.keys = self.keys[mid:]
        right.values = self.values[mid:]
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        return right, right.keys[0]

    def split_internal(self) -> Tuple["_BTreeNode", Any]:
        """Split a full internal node; return (new_right_node, promoted_key)."""
        mid = len(self.keys) // 2
        promoted = self.keys[mid]
        right = _BTreeNode(self.order, leaf=False)
        right.keys = self.keys[mid + 1:]
        right.children = self.children[mid + 1:]
        self.keys = self.keys[:mid]
        self.children = self.children[:mid + 1]
        return right, promoted


class BTree:
    """
    In-memory B-Tree.

    Keys can be any comparable Python value (int, str, or tuple for composite).
    Each key maps to a *list* of RecordIDs (to handle duplicates naturally).
    """

    def __init__(self, order: int = 64):
        self.order = max(order, 4)
        self.root = _BTreeNode(self.order, leaf=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, key, rid):
        result = self._insert(self.root, key, rid)
        if result is not None:
            right_child, promoted_key = result
            new_root = _BTreeNode(self.order, leaf=False)
            new_root.keys = [promoted_key]
            new_root.children = [self.root, right_child]
            self.root = new_root

    def search(self, key) -> List[Any]:
        return self._search(self.root, key)

    # ------------------------------------------------------------------
    # Internal recursion
    # ------------------------------------------------------------------
    def _insert(self, node: _BTreeNode, key, rid) -> Optional[Tuple[_BTreeNode, Any]]:
        if node.leaf:
            node._leaf_insert(key, rid)
            if node.is_full():
                return node.split_leaf()
            return None
        else:
            # Find the appropriate child
            child_idx = len(node.keys)
            for i, k in enumerate(node.keys):
                if key < k:
                    child_idx = i
                    break
            result = self._insert(node.children[child_idx], key, rid)
            if result is not None:
                right_child, promoted_key = result
                node._internal_insert(promoted_key, node.children[child_idx], right_child)
                if node.is_full():
                    return node.split_internal()
            return None

    def _search(self, node: _BTreeNode, key) -> List[Any]:
        if node.leaf:
            return node._leaf_search(key)
        for i, k in enumerate(node.keys):
            if key < k:
                return self._search(node.children[i], key)
        return self._search(node.children[-1], key)


# ---------------------------------------------------------------------------
# Index wrappers
# ---------------------------------------------------------------------------

class BTreeIndex:
    """
    Single-field B-Tree index.

    Parameters
    ----------
    tx           : active Transaction (not used at construction; kept for API parity)
    index_name   : logical name of this index
    key_type     : 'int' or 'str'
    key_length   : byte length of the field (informational only)
    """

    def __init__(self, tx, index_name: str, key_type: str, key_length: int):
        self.index_name = index_name
        self.key_type = key_type
        self.key_length = key_length
        self._btree = BTree(order=64)

    def insert(self, key_value, record_id: RecordID):
        self._btree.insert(key_value, record_id)

    def search(self, key_value) -> List[RecordID]:
        return self._btree.search(key_value)

    def close(self):
        pass  # in-memory — nothing to close


class CompositeIndex:
    """
    Multi-field B-Tree index keyed on *tuples* of field values.

    Parameters
    ----------
    tx             : active Transaction
    index_name     : logical name
    field_names    : tuple of field names, e.g. ('sec_semester', 'sec_year')
    field_types    : tuple of types,      e.g. ('str', 'int')
    field_lengths  : tuple of lengths,    e.g. (10, 4)
    """

    def __init__(self, tx, index_name: str,
                 field_names: Tuple[str, ...],
                 field_types: Tuple[str, ...],
                 field_lengths: Tuple[int, ...]):
        self.index_name = index_name
        self.field_names = field_names
        self.field_types = field_types
        self.field_lengths = field_lengths
        self._btree = BTree(order=64)

    def insert(self, field_values: Tuple, record_id: RecordID):
        """field_values must be a tuple in the same order as field_names."""
        self._btree.insert(field_values, record_id)

    def search(self, field_values: Tuple) -> List[RecordID]:
        return self._btree.search(field_values)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Index-driven scan
# ---------------------------------------------------------------------------

class IndexScan:
    """
    Replaces a full TableScan with an index lookup.

    For a single-field index the search_key is a scalar value.
    For a CompositeIndex the search_key is a tuple.

    Usage pattern mirrors TableScan / SelectScan so it can be dropped in
    wherever a scan is expected by the planner layer.
    """

    def __init__(self, table_scan: TableScan, index, search_key):
        self.table_scan = table_scan
        self.index = index
        self.search_key = search_key
        # Pre-fetch all matching RecordIDs up front
        self._rids: List[RecordID] = index.search(search_key)
        self._pos = -1

    # ------------------------------------------------------------------
    # Scan interface
    # ------------------------------------------------------------------
    def nextRecord(self) -> bool:
        self._pos += 1
        if self._pos >= len(self._rids):
            return False
        rid = self._rids[self._pos]
        self.table_scan.moveToRecordID(rid)
        return True

    def beforeFirst(self):
        self._pos = -1

    def getInt(self, field_name: str) -> int:
        return self.table_scan.getInt(field_name)

    def getString(self, field_name: str) -> str:
        return self.table_scan.getString(field_name)

    def getVal(self, field_name: str):
        return self.table_scan.getVal(field_name)

    def hasField(self, field_name: str) -> bool:
        return self.table_scan.hasField(field_name)

    def closeRecordPage(self):
        self.table_scan.closeRecordPage()


# ---------------------------------------------------------------------------
# Index plan node (wraps IndexScan as a Plan)
# ---------------------------------------------------------------------------

class IndexPlan:
    """
    A Plan that opens an IndexScan instead of a full TableScan.

    blocksAccessed() returns 1 as a heuristic — index lookup touches very
    few blocks compared with a full table scan.
    """

    def __init__(self, tx: Transaction, table_name: str, layout: Layout,
                 index, search_key, table_stat: dict):
        self.tx = tx
        self.table_name = table_name
        self.layout = layout
        self.index = index
        self.search_key = search_key
        self._stat = table_stat
        self._schema = layout.schema

    def open(self):
        ts = TableScan(self.tx, self.table_name, self.layout)
        return IndexScan(ts, self.index, self.search_key)

    def blocksAccessed(self) -> int:
        return 1   # index lookup ≈ O(log n) blocks — treat as 1 for cost model

    def recordsOutput(self) -> int:
        # rough estimate: total rows / distinct values
        dv = self._stat.get('distinctValues', 1) or 1
        return max(1, int(self._stat.get('recordsOutput', 1) / dv))

    def distinctValues(self, field_name: str):
        return 1   # after equality filter, only 1 distinct value remains

    def plan_schema(self) -> Schema:
        return self._schema


# ---------------------------------------------------------------------------
# Helper: predicate analysis
# ---------------------------------------------------------------------------

def _classify_terms(query_data: dict, table_names: List[str]):
    """
    Partition query predicate terms into:
      - single_table_terms  : {table_name -> [Term, ...]}   (selection pushdown)
      - join_terms          : list of (Term, table_name_a, table_name_b)
      - remaining_terms     : Terms that could not be classified

    A term is classified as a single-table term if both its LHS and RHS
    field references (if any) belong to the same table.

    A term is classified as a join term if its LHS and RHS reference
    fields from exactly two different tables.
    """
    pred: Predicate = query_data['predicate']
    # Build field → table mapping
    field_to_table: Dict[str, str] = {}
    for tname in table_names:
        for fname in query_data.get('_layouts', {}).get(tname, {}).keys():
            field_to_table[fname] = tname

    single_table_terms: Dict[str, List[Term]] = {t: [] for t in table_names}
    join_terms: List[Tuple[Term, str, str]] = []
    remaining_terms: List[Term] = []

    for term in pred.terms:
        lhs: Expression = term.lhs
        rhs: Expression = term.rhs

        lhs_field = lhs.exp_value if not isinstance(lhs.exp_value, Constant) else None
        rhs_field = rhs.exp_value if not isinstance(rhs.exp_value, Constant) else None

        lhs_table = field_to_table.get(lhs_field) if lhs_field else None
        rhs_table = field_to_table.get(rhs_field) if rhs_field else None

        if lhs_table is None and rhs_table is None:
            # constant = constant, unlikely but handle gracefully
            remaining_terms.append(term)
        elif lhs_table is None:
            # constant op field
            single_table_terms[rhs_table].append(term)
        elif rhs_table is None:
            # field op constant
            single_table_terms[lhs_table].append(term)
        elif lhs_table == rhs_table:
            single_table_terms[lhs_table].append(term)
        else:
            join_terms.append((term, lhs_table, rhs_table))

    return single_table_terms, join_terms, remaining_terms


def _make_predicate(terms: List[Term]) -> Predicate:
    p = Predicate()
    p.terms = list(terms)
    return p


def _tables_in_schema(schema: Schema, field_to_table: Dict[str, str]) -> set:
    return {field_to_table[f] for f in schema.field_info if f in field_to_table}


# ---------------------------------------------------------------------------
# BetterQueryPlanner
# ---------------------------------------------------------------------------

class BetterQueryPlanner:
    """
    Optimized query planner:
      1. Selection pushdown — wrap each TablePlan with a SelectPlan for
         predicates that reference only that table's fields.
      2. Join reordering — greedily join the table that has a connecting
         join condition to the current result set (or failing that, the
         smallest table by estimated block count).
      3. Remaining join conditions are applied immediately after each join.
      4. Final projection.
    """

    def __init__(self, mm: MetadataMgr):
        self.mm = mm

    def createPlan(self, tx: Transaction, query_data: dict):
        table_names: List[str] = query_data['tables']

        # ------------------------------------------------------------------
        # Build layouts so _classify_terms can map fields to tables
        # ------------------------------------------------------------------
        layouts: Dict[str, Layout] = {}
        for tname in table_names:
            layouts[tname] = self.mm.getLayout(tx, tname)
        query_data['_layouts'] = {
            tname: layouts[tname].schema.field_info
            for tname in table_names
        }

        # ------------------------------------------------------------------
        # Build a TablePlan for every table
        # ------------------------------------------------------------------
        table_plans: Dict[str, TablePlan] = {
            tname: TablePlan(tx, tname, self.mm)
            for tname in table_names
        }

        # ------------------------------------------------------------------
        # Classify predicates
        # ------------------------------------------------------------------
        single_terms, join_terms, remaining_terms = _classify_terms(
            query_data, table_names
        )

        # field → table mapping (used later)
        field_to_table: Dict[str, str] = {}
        for tname in table_names:
            for fname in layouts[tname].schema.field_info:
                field_to_table[fname] = tname

        # ------------------------------------------------------------------
        # Selection pushdown
        # ------------------------------------------------------------------
        pushed_plans: Dict[str, any] = {}
        for tname in table_names:
            plan = table_plans[tname]
            terms = single_terms.get(tname, [])
            if terms:
                plan = SelectPlan(plan, _make_predicate(terms))
            pushed_plans[tname] = plan

        # ------------------------------------------------------------------
        # Join reordering
        # ------------------------------------------------------------------
        remaining_tables = list(table_names)
        # Sort by estimated cost (blocks * records) ascending as starting point
        remaining_tables.sort(
            key=lambda t: pushed_plans[t].blocksAccessed()
        )

        # Start with the cheapest table
        current_plan = pushed_plans[remaining_tables[0]]
        joined_tables = {remaining_tables[0]}
        remaining_tables = remaining_tables[1:]

        # Track which join terms have been applied
        applied_join_terms = set()

        while remaining_tables:
            # Find the next table that connects to our current result via a join term
            best_table = None
            best_score = None
            best_applicable = []

            for candidate in remaining_tables:
                # Collect join terms that connect candidate to already-joined tables
                applicable = []
                for idx, (term, ta, tb) in enumerate(join_terms):
                    if idx in applied_join_terms:
                        continue
                    if (ta == candidate and tb in joined_tables) or \
                       (tb == candidate and ta in joined_tables):
                        applicable.append(idx)

                # Score: prefer tables with join conditions (applicable > 0), then by cost
                score = (
                    0 if applicable else 1,
                    pushed_plans[candidate].blocksAccessed()
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_table = candidate
                    best_applicable = applicable

            # Perform the join
            current_plan = ProductPlan(current_plan, pushed_plans[best_table])
            joined_tables.add(best_table)
            remaining_tables.remove(best_table)

            # Apply join conditions that now connect fully
            for idx in best_applicable:
                term, ta, tb = join_terms[idx]
                applied_join_terms.add(idx)
                join_pred = _make_predicate([term])
                current_plan = SelectPlan(current_plan, join_pred)

        # ------------------------------------------------------------------
        # Apply any leftover join or remaining terms
        # ------------------------------------------------------------------
        leftover = []
        for idx, (term, ta, tb) in enumerate(join_terms):
            if idx not in applied_join_terms:
                leftover.append(term)
        leftover.extend(remaining_terms)
        if leftover:
            current_plan = SelectPlan(current_plan, _make_predicate(leftover))

        # ------------------------------------------------------------------
        # Project the required fields
        # ------------------------------------------------------------------
        return ProjectPlan(current_plan, *query_data['fields'])


# ---------------------------------------------------------------------------
# IndexQueryPlanner
# ---------------------------------------------------------------------------

class IndexQueryPlanner:
    """
    Query planner that substitutes IndexScan for TableScan wherever an index
    covers an equality predicate (field = constant) in the WHERE clause.

    If a better_planner is supplied (BetterQueryPlanner), it is used for
    join ordering and selection pushdown; otherwise the original join order
    from the query is preserved.

    The index lookup happens at the TablePlan level: for each table, if a
    single-table equality term of the form  field = constant  (or
    constant = field) exists and an index covers that field, the table's
    plan is replaced with an IndexPlan.

    For composite indexes (keyed on tuples), all component equality terms
    must be present in the predicate for the composite index to be chosen.
    """

    def __init__(self, mm: MetadataMgr, indexes: Dict, better_planner=None):
        """
        Parameters
        ----------
        mm             : MetadataMgr
        indexes        : {table_name: {field_key: IndexObject}}
                         field_key is str for BTreeIndex, tuple for CompositeIndex
        better_planner : optional BetterQueryPlanner for join optimisation
        """
        self.mm = mm
        self.indexes = indexes or {}
        self.better_planner = better_planner

    def createPlan(self, tx: Transaction, query_data: dict):
        table_names: List[str] = query_data['tables']

        # ------------------------------------------------------------------
        # Build layouts for field-to-table mapping
        # ------------------------------------------------------------------
        layouts: Dict[str, Layout] = {}
        for tname in table_names:
            layouts[tname] = self.mm.getLayout(tx, tname)
        query_data['_layouts'] = {
            tname: layouts[tname].schema.field_info
            for tname in table_names
        }

        # ------------------------------------------------------------------
        # Classify predicate terms
        # ------------------------------------------------------------------
        single_terms, join_terms, remaining_terms = _classify_terms(
            query_data, table_names
        )

        # ------------------------------------------------------------------
        # For each table: decide between IndexPlan or TablePlan (+ pushdown)
        # ------------------------------------------------------------------
        base_plans: Dict[str, any] = {}

        for tname in table_names:
            layout = layouts[tname]
            stat = self.mm.getStatInfo(tx, tname, layout)
            table_index_map = self.indexes.get(tname, {})
            terms_for_table = single_terms.get(tname, [])

            # --- Try composite indexes first (more selective) ---
            chosen_index = None
            chosen_key = None
            used_term_indices: List[int] = []

            for field_key, idx in table_index_map.items():
                if not isinstance(field_key, tuple):
                    continue  # single-field index — handled below
                # Check all component fields have equality constants in pred
                values = []
                term_idxs = []
                for comp_field in field_key:
                    found = False
                    for ti, term in enumerate(terms_for_table):
                        const_val = _extract_constant(term, comp_field)
                        if const_val is not None:
                            values.append(const_val)
                            term_idxs.append(ti)
                            found = True
                            break
                    if not found:
                        break
                if len(values) == len(field_key):
                    chosen_index = idx
                    chosen_key = tuple(values)
                    used_term_indices = term_idxs
                    break

            # --- Try single-field indexes ---
            if chosen_index is None:
                for field_key, idx in table_index_map.items():
                    if isinstance(field_key, tuple):
                        continue
                    for ti, term in enumerate(terms_for_table):
                        const_val = _extract_constant(term, field_key)
                        if const_val is not None:
                            chosen_index = idx
                            chosen_key = const_val
                            used_term_indices = [ti]
                            break
                    if chosen_index is not None:
                        break

            if chosen_index is not None:
                # Build IndexPlan, then push any remaining single-table terms
                plan = IndexPlan(tx, tname, layout, chosen_index, chosen_key, stat)
                leftover_terms = [
                    t for i, t in enumerate(terms_for_table)
                    if i not in used_term_indices
                ]
                if leftover_terms:
                    plan = SelectPlan(plan, _make_predicate(leftover_terms))
            else:
                # No usable index — fall back to TablePlan + pushdown
                plan = TablePlan(tx, tname, self.mm)
                if terms_for_table:
                    plan = SelectPlan(plan, _make_predicate(terms_for_table))

            base_plans[tname] = plan

        # ------------------------------------------------------------------
        # Join ordering
        # ------------------------------------------------------------------
        if self.better_planner is not None:
            # Delegate join ordering to BetterQueryPlanner but inject our
            # already-built base_plans by monkeypatching createPlan to use them.
            return self._join_with_better(tx, query_data, base_plans, join_terms,
                                          remaining_terms, table_names)
        else:
            return self._join_original_order(query_data, base_plans, join_terms,
                                             remaining_terms, table_names)

    # ------------------------------------------------------------------
    # Internal join helpers
    # ------------------------------------------------------------------
    def _join_original_order(self, query_data, base_plans, join_terms,
                              remaining_terms, table_names):
        """Keep original table order; apply join conditions after each step."""
        applied = set()
        joined_tables = {table_names[0]}
        current_plan = base_plans[table_names[0]]

        for tname in table_names[1:]:
            current_plan = ProductPlan(current_plan, base_plans[tname])
            joined_tables.add(tname)

            # Apply any join terms that now fully connect
            for idx, (term, ta, tb) in enumerate(join_terms):
                if idx in applied and ta in joined_tables and tb in joined_tables:
                    continue
                if idx not in applied and ta in joined_tables and tb in joined_tables:
                    applied.add(idx)
                    current_plan = SelectPlan(current_plan, _make_predicate([term]))

        # Remaining
        leftover = [t for i, (t, _, __) in enumerate(join_terms) if i not in applied]
        leftover.extend(remaining_terms)
        if leftover:
            current_plan = SelectPlan(current_plan, _make_predicate(leftover))

        return ProjectPlan(current_plan, *query_data['fields'])

    def _join_with_better(self, tx, query_data, base_plans, join_terms,
                           remaining_terms, table_names):
        """Reorder joins using the same greedy heuristic as BetterQueryPlanner."""
        remaining = sorted(
            list(table_names),
            key=lambda t: base_plans[t].blocksAccessed()
        )

        current_plan = base_plans[remaining[0]]
        joined_tables = {remaining[0]}
        remaining = remaining[1:]
        applied = set()

        while remaining:
            best_table = None
            best_score = None
            best_applicable = []

            for candidate in remaining:
                applicable = []
                for idx, (term, ta, tb) in enumerate(join_terms):
                    if idx in applied:
                        continue
                    if (ta == candidate and tb in joined_tables) or \
                       (tb == candidate and ta in joined_tables):
                        applicable.append(idx)
                score = (0 if applicable else 1, base_plans[candidate].blocksAccessed())
                if best_score is None or score < best_score:
                    best_score = score
                    best_table = candidate
                    best_applicable = applicable

            current_plan = ProductPlan(current_plan, base_plans[best_table])
            joined_tables.add(best_table)
            remaining.remove(best_table)

            for idx in best_applicable:
                term, ta, tb = join_terms[idx]
                applied.add(idx)
                current_plan = SelectPlan(current_plan, _make_predicate([term]))

        leftover = [t for i, (t, _, __) in enumerate(join_terms) if i not in applied]
        leftover.extend(remaining_terms)
        if leftover:
            current_plan = SelectPlan(current_plan, _make_predicate(leftover))

        return ProjectPlan(current_plan, *query_data['fields'])


# ---------------------------------------------------------------------------
# Utility: extract a constant from an equality Term
# ---------------------------------------------------------------------------

def _extract_constant(term: Term, field_name: str):
    """
    If term is of the form  field_name = <constant>  or  <constant> = field_name,
    return the constant value; otherwise return None.
    """
    from RelationalOp import Constant as RConstant
    lhs: Expression = term.lhs
    rhs: Expression = term.rhs

    lhs_is_field = not isinstance(lhs.exp_value, RConstant)
    rhs_is_field = not isinstance(rhs.exp_value, RConstant)

    if lhs_is_field and not rhs_is_field:
        if lhs.exp_value == field_name:
            return rhs.exp_value.const_value
    elif rhs_is_field and not lhs_is_field:
        if rhs.exp_value == field_name:
            return lhs.exp_value.const_value
    return None


# ---------------------------------------------------------------------------
# create_indexes
# ---------------------------------------------------------------------------

def create_indexes(db, tx: Transaction,
                   index_defs: Optional[Dict] = None,
                   composite_index_defs: Optional[Dict] = None) -> Dict:
    """
    Build and populate all indexes by scanning each table once.

    Parameters
    ----------
    db                   : BenchmarkDB (has .mm : MetadataMgr)
    tx                   : Transaction
    index_defs           : {table_name: [(field_name, field_type, field_length), ...]}
    composite_index_defs : {table_name: [((field_names,...), (field_types,...), (field_lengths,...))]}

    Returns
    -------
    dict {table_name: {field_key: IndexObject}}
      field_key is str for BTreeIndex, tuple for CompositeIndex
    """
    index_defs = index_defs or {}
    composite_index_defs = composite_index_defs or {}

    # ------------------------------------------------------------------
    # 1. Instantiate all index objects
    # ------------------------------------------------------------------
    all_indexes: Dict[str, Dict] = {}

    for table_name, fields in index_defs.items():
        all_indexes.setdefault(table_name, {})
        for field_name, field_type, field_length in fields:
            idx_name = f"idx_{table_name}_{field_name}"
            all_indexes[table_name][field_name] = BTreeIndex(
                tx, idx_name, field_type, field_length
            )

    for table_name, comp_defs in composite_index_defs.items():
        all_indexes.setdefault(table_name, {})
        for field_names_tuple, field_types_tuple, field_lengths_tuple in comp_defs:
            idx_name = "idx_{}_{}".format(table_name, "_".join(field_names_tuple))
            key = tuple(field_names_tuple)
            all_indexes[table_name][key] = CompositeIndex(
                tx, idx_name, field_names_tuple, field_types_tuple, field_lengths_tuple
            )

    # ------------------------------------------------------------------
    # 2. Populate: scan each table once and insert into all its indexes
    # ------------------------------------------------------------------
    for table_name, idx_map in all_indexes.items():
        if not idx_map:
            continue

        layout = db.mm.getLayout(tx, table_name)
        ts = TableScan(tx, table_name, layout)

        while ts.nextRecord():
            rid = ts.currentRecordID()

            for field_key, idx in idx_map.items():
                if isinstance(field_key, tuple):
                    # Composite index
                    vals = tuple(ts.getVal(f) for f in field_key)
                    idx.insert(vals, rid)
                else:
                    # Single-field index
                    val = ts.getVal(field_key)
                    idx.insert(val, rid)

        ts.closeRecordPage()

    return all_indexes
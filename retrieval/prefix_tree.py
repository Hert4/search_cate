from typing import List, Optional, Tuple
from models.product import Product
from utils.text_normalization import normalize_text


class PrefixTreeNode:
    def __init__(self, token: str):
        self.token = token
        self.children = {}
        self.products = []
    
    def add_child(self, token: str):
        if token not in self.children:
            self.children[token] = PrefixTreeNode(token)
        return self.children[token]


def build_prefix_tree(products: List[Product]) -> PrefixTreeNode:
    """Xây dựng prefix tree"""
    root = PrefixTreeNode("ROOT")
    
    for product in products:
        current = root
        full_path = " ".join(product.category_path)
        tokens = normalize_text(full_path)
        
        # Loại bỏ duplicate liên tiếp
        unique_tokens = []
        for token in tokens:
            if not unique_tokens or token != unique_tokens[-1]:
                unique_tokens.append(token)
        
        for token in unique_tokens:
            current = current.add_child(token)
        
        current.products.append(product)
    
    return root


def find_matching_node(tree: PrefixTreeNode, query: str) -> Optional[PrefixTreeNode]:
    """Tìm node sâu nhất match với query"""
    query_tokens = normalize_text(query)
    
    current = tree
    deepest_with_products = None
    
    for token in query_tokens:
        if token in current.children:
            current = current.children[token]
            if current.products:
                deepest_with_products = current
        else:
            break
    
    return deepest_with_products



def find_matching_node_bottom_up(tree: PrefixTreeNode, query: str) -> Optional[PrefixTreeNode]:
    """
    Tìm node match với query bằng cách duyệt từ DƯỚI LÊN (bottom-up)
    Tìm xem query có chứa các token ở tầng sâu nhất không, rồi mới check tầng trên
    """
    query_tokens = set(normalize_text(query))  # Dùng set để tra cứu nhanh
    
    def collect_all_nodes(node: PrefixTreeNode, depth: int = 0) -> List[Tuple[PrefixTreeNode, int]]:
        """Thu thập tất cả nodes với độ sâu của chúng"""
        nodes = [(node, depth)]
        for child in node.children.values():
            nodes.extend(collect_all_nodes(child, depth + 1))
        return nodes
    
    # Thu thập tất cả nodes và sort theo độ sâu (sâu nhất trước)
    all_nodes = collect_all_nodes(tree)
    all_nodes.sort(key=lambda x: x[1], reverse=True)  # Sort giảm dần theo depth
    
    # Duyệt từ node sâu nhất
    for node, depth in all_nodes:
        if not node.products:  # Bỏ qua node không có products
            continue
        
        # Lấy path từ root đến node này
        path_tokens = []
        current = node
        
        # Reconstruct path (cần parent pointer, hoặc dùng cách khác)
        # Vì không có parent pointer, ta sẽ dùng cách đơn giản hơn:
        # Check xem token của node này có trong query không
        if node.token in query_tokens:
            return node
    
    return None


def find_matching_node_bottom_up_v2(tree: PrefixTreeNode, query: str) -> Optional[PrefixTreeNode]:
    """
    Version 2: Duyệt từ dưới lên bằng cách check FULL PATH
    Tìm node có path khớp nhiều nhất với query, ưu tiên node sâu
    """
    query_tokens = set(normalize_text(query))
    
    def get_node_path_and_depth(node: PrefixTreeNode, 
                                 current_path: List[str], 
                                 depth: int) -> List[Tuple[PrefixTreeNode, List[str], int]]:
        """Recursively collect nodes with their paths and depths"""
        results = []
        
        if node.products:  # Chỉ lấy node có products
            results.append((node, current_path.copy(), depth))
        
        for token, child in node.children.items():
            results.extend(get_node_path_and_depth(
                child, 
                current_path + [token], 
                depth + 1
            ))
        
        return results
    
    # Thu thập tất cả nodes với path và depth
    all_nodes_with_info = get_node_path_and_depth(tree, [], 0)
    
    if not all_nodes_with_info:
        return None
    
    # Tính score cho mỗi node
    scored_nodes = []
    for node, path, depth in all_nodes_with_info:
        path_tokens = set(path)
        
        # Score = số token match giữa query và path
        matches = len(query_tokens & path_tokens)
        
        # Ưu tiên node sâu hơn khi số match bằng nhau
        score = (matches, depth)
        
        scored_nodes.append((node, score, path))
    
    # Sort: match nhiều nhất, depth sâu nhất
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Trả về node tốt nhất (nếu có ít nhất 1 token match)
    if scored_nodes[0][1][0] > 0:  # matches > 0
        return scored_nodes[0][0]
    
    return None


def find_matching_node_bottom_up_v3(tree: PrefixTreeNode, query: str) -> Optional[PrefixTreeNode]:
    """
    Version 3: Duyệt từ cuối query ngược lên
    Tìm xem token cuối của query có match với node nào không, rồi mới check các token trước
    """
    query_tokens = normalize_text(query)
    
    if not query_tokens:
        return None
    
    def find_nodes_by_token(node: PrefixTreeNode, target_token: str, 
                            current_path: List[str]) -> List[Tuple[PrefixTreeNode, List[str]]]:
        """Tìm tất cả nodes có token = target_token"""
        results = []
        
        if node.token == target_token and node.products:
            results.append((node, current_path.copy()))
        
        for token, child in node.children.items():
            results.extend(find_nodes_by_token(child, target_token, current_path + [token]))
        
        return results
    
    # Duyệt từ token cuối của query ngược về đầu
    for i in range(len(query_tokens) - 1, -1, -1):
        target_token = query_tokens[i]
        
        # Tìm nodes có token này
        matching_nodes = find_nodes_by_token(tree, target_token, [])
        
        if matching_nodes:
            # Verify các token trước đó có trong path không
            best_node = None
            best_match_count = 0
            
            for node, path in matching_nodes:
                path_set = set(path)
                
                # Đếm số token trước target_token có trong path
                match_count = sum(1 for t in query_tokens[:i] if t in path_set)
                
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_node = node
            
            if best_node:
                return best_node
    
    return None

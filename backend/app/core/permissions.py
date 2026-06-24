"""
RBAC Permission Engine

Pre-filter logic for RAG retrieval. Injects permission assertions into
vector search (Milvus filter expression) and keyword search (BM25 chunk filter).

Design:
- Admin/superuser: sees everything, no filter
- Manager: sees own departments + lower clearance
- Member: sees own departments + public/internal only

The permission filter is a Milvus expression string like:
  (security_level <= 2) AND (department in ["engineering", "product"])
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

from app.models.user import User, ROLE_ADMIN, ROLE_MANAGER, ROLE_MEMBER


@dataclass
class PermissionFilter:
    """Permission constraints for a search query."""
    max_security: int                      # max allowed security level
    departments: List[str]                  # allowed departments
    kb_ids: Optional[List[str]] = None     # explicitly allowed KBs (overrides)

    @property
    def milvus_expr(self) -> str:
        """Build Milvus pre-filter expression."""
        parts = [f"security_level <= {self.max_security}"]
        if self.departments:
            deps = [d for d in self.departments if d and d != "_"]
            if deps:
                dlist = ", ".join(f'"{d}"' for d in deps)
                parts.append(f"department in [{dlist}]")
        return " AND ".join(parts)

    @property
    def bm25_expr(self) -> str:
        """Same as milvus_expr but used for logging/BM25 chunk filter."""
        return self.milvus_expr

    def chunk_allowed(self, chunk: Dict) -> bool:
        """Check if a single chunk passes permission filter (for BM25)."""
        sl = chunk.get("security_level", 0)
        dep = chunk.get("department", "")
        if sl > self.max_security:
            return False
        if self.departments and dep and dep not in self.departments:
            return False
        return True


def build_permission_filter(
    user: Optional[User],
    target_kb_ids: Optional[List[str]] = None,
) -> PermissionFilter:
    """
    Build permission filter for a user.

    Admin/superuser: no restrictions.
    Others: filtered by clearance + departments.
    """
    if user is None:
        # Anonymous: public + internal, all departments (dev default)
        return PermissionFilter(max_security=1, departments=[], kb_ids=target_kb_ids)

    if user.can_access_all:
        # Admin sees everything
        return PermissionFilter(max_security=999, departments=[], kb_ids=target_kb_ids)

    return PermissionFilter(
        max_security=user.clearance_level,
        departments=user.departments or [],
        kb_ids=target_kb_ids,
    )

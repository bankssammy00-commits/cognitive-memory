from app.models.entity import Entity
from app.models.association import Association


def test_entity():

    entity = Entity(
        id="entity_001",
        name="Supplier X",
        entity_type="supplier",
        aliases=["SupplierX"],
        confidence=0.95,
    )

    assert entity.name == "Supplier X"
    assert entity.entity_type == "supplier"


def test_association():

    association = Association(
        id="association_001",
        source_id="memory_001",
        target_id="memory_002",
        relationship_type="supports",
        strength=0.85,
    )

    assert association.relationship_type == "supports"
    assert association.strength == 0.85
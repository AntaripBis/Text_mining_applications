from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Introductory Fast API application")

class Category(Enum):
    TOOLS = "tools"
    CONSUMABLES = "consumables"

class Item(BaseModel):
    name: str = Field(description="Name of the item")
    price: float = Field(description="Price of the item")
    stock: int = Field(description="Number of item")
    id: int = Field(description="ID of the item")
    category: Category = Field(description="category of the item")


items = {
    0: Item(name="Hammer",price=9.99,stock=20, id=0, category=Category.TOOLS),
    1: Item(name="Pliers",price=5.99, stock=20, id=1,category=Category.TOOLS),
    2: Item(name="Nails",price=1.99,stock=2000,id=2,category=Category.CONSUMABLES)
}

@app.get("/")
def index() -> dict[str,dict[int,Item]]:
    return {"items":items}

@app.get("/items/{item_id}")
def get_item_by_id(item_id: int) -> Item:
    if item_id not in items:
        raise HTTPException(status_code=404,detail=f"item with {item_id} is not found")
    
    return items[item_id]

Selection = dict[str, str | float | int | Category | None]

@app.get("/items/")
def get_item_by_parameters(name: str | None=None,price: float | None = None,
                           stock: int | None = None, 
                           category: Category | None = None) -> dict[str, Selection | list[Item]]:
    def check_item(item: Item) -> bool:
        return all(
            [
                name is None or item.name == name,
                price is None or item.price == price,
                stock is None or item.stock == stock,
                category is None or item.category == category
            ]
        )
    
    selection = [item for item in items.values() if check_item(item)]
    
    return {
        "query":{"name":name,"price":price,"stock":stock,"category":category},
        "selection":selection
    } 

@app.post("/")
def add_item(item: Item) -> dict[str,Item]:
    if item.id in items:
        raise HTTPException(status_code=400,detail=f"item with id {item.id} is already present")
    
    items[item.id] = item
    return {"added":item}

@app.put("/update/{item_id}")
def update_item(item_id: int,name: str | None = None,price: float | None = None,
                stock: int | None = None) -> dict[str,Item]:
    
    if item_id not in items:
        raise HTTPException(status_code=404,detail=f"item with id {item_id} does not exist")
    
    if all([x is None for x in [name, price, stock]]):
        raise HTTPException(status_code=400, detail="at least name, price or stock has to be non-null")

    if name is not None:
        items[item_id].name = name

    items[item_id].price = items[item_id].price if price is None else price
    items[item_id].stock = items[item_id].stock if stock is None else stock

    return {"updated":items[item_id]}

@app.delete("/delete/{item_id}")
def delete_item(item_id: int) -> dict[str,Item]:
    if item_id not in items:
        raise HTTPException(status_code=404,detail=f"item with id {item_id} not found")
    
    item = items.pop(item_id)
    return {"removed":item}
    

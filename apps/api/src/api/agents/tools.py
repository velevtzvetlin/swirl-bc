import openai
import cohere
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchAny, Prefetch, Document,MatchValue
from qdrant_client import models
from langchain_core.tools import tool

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np


@traceable(
    name="embed_query",
    run_type="embedding", # tells langsmith we are actually running embedding text -> specific set of vectors
    metadata={
        "ls_provider": "openai", 
        "ls_model_name": "text-embedding-3-small"
    } ## langsmith needs this to calculate cost of runs
)
def get_embedding(text):
    response = openai.embeddings.create( # was client before
        input=text,
        model="text-embedding-3-small"
    )
    
    current_run  = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_items_data",
    run_type="retriever"
)
def retrieve_items_data(query, qdrant_client, k=5, hybrid=True): # 5 most similar items to users query
    embedding = get_embedding(query) # so we are actually creating related vector here
    
    if hybrid:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=embedding,
                    using='text-embedding-3-small', #name of the vector in the collection
                    limit=20
                ), 
                Prefetch(
                    query=Document(text=query, model='qdrant/bm25'),
                    using='bm25', #name of the vector in the collection
                    limit=20
                )
            ],
            # can set specific weights on how much importance we put on dense embeddings and bm25 lexical retrievals
            # in this case dense vector retrival is weighted as 3 times more important than bm25
            # if user queries are more keyword-based, you might want to increase the weight of bm25
            # if more contextual then you might want to increase the weight of the dense vector retrieval
            # weights can be added as params to the retrieve data function
            query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])), # how you fuse the results....
            limit=k # once fused we are returning top k results
        )
    else:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=embedding,
            using='text-embedding-3-small',
            limit=k
        )
    
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload.get("preprocessed_description"))
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload.get("average_rating"))
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

@traceable(
    name="rerank_data",
    run_type="tool"
)
def rerank_data(query, context, topk_k=5):
    cohere_client = cohere.ClientV2()
    # so we pass the query and the array of preprocessed_description name, desc context chunks
    response = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=context['retrieved_context'],
        top_n=topk_k
    )
    order = [result.index for result in response.results]
    # re order the original context
    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [context["retrieved_context_ratings"][i] for i in order]
    }
    
@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context   

@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """
        Search available products and return the top k matching inventory items.

        Expand the customer's question into 1–5 concise search statements and issue them in parallel in a single turn. Each statement covers one distinct product or attribute; no two may express the same intent. Use natural product-description language. If no brand or model is specified, search broadly rather than refusing.

        "Earphones for me and a waterproof speaker"
        -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
        -> "Insulated winter jacket" | "Hiking outerwear for cold weather"

        Before calling, check what earlier calls in this conversation already returned. Search only for what is missing; results already retrieved remain valid and must not be fetched again.

        Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.

        Returns:
        A string of the top k available products, each prefixed with its ID and average rating.
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network
    retrievedContext = retrieve_items_data(
        query, 
        qdrant_client, 
        k=20
    )
    # if we set rerank to True, we re-order the retrieved context based on the reranker results
    retrievedContext = rerank_data(query, retrievedContext, topk_k=top_k)
    formatted_context = process_context(retrievedContext)
    return formatted_context


# reviews retrieval metadata tool
@traceable(
    name="retrieve_prefiltered_reviews_data",
    run_type="retriever"
)
def retrieve_prefiltered_reviews_data(query, parent_asins, qdrant_client, top_k=5): # 5 most similar items to users query
    embedding = get_embedding(query) # so we are actually creating related vector here
    
    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=embedding,
                using='text-embedding-3-small', #name of the vector in the collection
                filter=Filter( 
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=parent_asins
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )
    
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload.get("preprocessed_data", ""))
        similarity_scores.append(result.score)
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_reviews_context(context):
    formatted_context = ""
    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, user review: {chunk}\n"
    return formatted_context   

@tool
def get_formatted_reviews_context(query: str, parent_asins: list[str], top_k: int = 5) -> str:
    """
        Get the top k reviews matching a query for a list of prefiltered items.

        Args:
            query: The query to get the top k reviews for
            item_list: The list of item IDs to prefilter for before running the query
            top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered

        Returns:
            A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network
    retrievedContext = retrieve_prefiltered_reviews_data(
        query,
        parent_asins,
        qdrant_client,
        top_k=top_k
    )
    # if we set rerank to True, we re-order the retrieved context based on the reranker results
    formatted_context = process_reviews_context(retrievedContext)
    return formatted_context


### Shopping cart tools

### add to shopping cart tool
@tool
## ads these items to the shopping cart .... 
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.

    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )

    conn.autocommit = True
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            print(item)
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url="http://qdrant:6333")

            dummy_vector = np.zeros(1536).tolist()
            res = qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                        ),
                        using="text-embedding-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1
            )
            print(res)
            payload = res.points[0].payload
            
            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = "USD"
            
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price
                FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone() # fetch one item and if it exists

            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity

                update_query = """
                UPDATE shopping_carts.shopping_cart_items
                SET
                    quantity = %s,
                    price = %s,
                    currency = %s,
                    product_image_url = COALESCE(%s, product_image_url)
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                RETURNING id, quantity, price
                """
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id, quantity, price, currency, product_image_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                cursor.execute(insert_query, (user_id, cart_id, product_id, quantity, price, currency, product_image_url))
    
    return f"Added {items} items to the shopping cart."

### get the shopping cart items tool

@tool
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """
    Retrieve all items in a user's shopping cart.

    Args:
        user_id: User identifier
        cart_id: Cart identifier

    Returns:
        list of dictionaries containing cart items
    """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        SELECT
            product_id, price, quantity,
            currency, product_image_url,
            (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s
        ORDER BY added_at DESC
        """  
        cursor.execute(query, (user_id, cart_id))
        
        return [row for row in cursor.fetchall()]
    
def get_shopping_cart_for_sse(user_id: str, cart_id: str) -> list[dict]:
    """
    Retrieve all items in a user's shopping cart.

    Args:
        user_id: User identifier
        cart_id: Cart identifier

    Returns:
        list of dictionaries containing cart items
    """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        SELECT
            product_id, price, quantity,
            currency, product_image_url,
            (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s
        ORDER BY added_at DESC
        """  
        cursor.execute(query, (user_id, cart_id))
        
        return [row for row in cursor.fetchall()]


#remove from shopping cart tool
@tool
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:
    """
    Remove an item completely from the shopping cart.

    Args:
        user_id: User identifier
        product_id: Product identifier to remove
        cart_id: Cart identifier

    Returns:
        Information about the removal of the item from the shopping cart.
    """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        DELETE FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """
        cursor.execute(query, (user_id, cart_id, product_id))

        return f"Removed {product_id} from the shopping cart." if cursor.rowcount > 0 else f"Item {product_id} not found in the shopping cart."
 
### Warehouse manager agent tools 
@tool   
def check_warehouse_availability(items: list[dict]) -> dict:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": []
            }

            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                # Reset warehouse categorization flags
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse['warehouse_id'],
                    "warehouse_name": warehouse['warehouse_name'],
                    "warehouse_location": warehouse['warehouse_location'],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False
                }

            for item in items:
                product_id = item['product_id']
                requested_quantity = item['quantity']

                # Check availability in this warehouse
                availability_query = """
                    SELECT product_id, total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                """
                cursor.execute(availability_query, (warehouse['warehouse_id'], product_id))
                inventory = cursor.fetchone()

                available_qty = inventory['available_quantity'] if inventory else 0

                item_detail = {
                    "product_id": product_id,
                    "requested": requested_quantity,
                    "available": available_qty,
                    "can_fulfill_completely": available_qty >= requested_quantity,
                    "can_fulfill_partially": available_qty > 0 and available_qty < requested_quantity
                }
                
                warehouse_details["items"].append(item_detail)

                # Track if warehouse can fulfill this item completely
                if available_qty < requested_quantity:
                    warehouse_can_fulfill_all = False

                # Track if warehouse has any availability for any item
                if available_qty > 0:
                    has_any_availability = True

                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })

                result["details"].append(warehouse_details)
            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item['product_id']
                requested_quantity = item['quantity']

                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                total_available_qty = total_available['total_available'] if total_available else 0

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append({
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "total_available_across_warehouses": total_available_qty,
                        "shortage": requested_quantity - total_available_qty
                    })
            
        result["can_fulfill_completely"] = len(result["warehouses_full_fulfillment"]) > 0 and len(result["unavailable_items"]) == 0
        return result

    finally:
        conn.close()

@tool
def reserve_warehouse_items(reservations: list[dict]) -> dict:
    """Reserve items from multiple warehouses in a single transaction.

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
            - warehouse_id: The warehouse to reserve from
            - product_id: The product to reserve
            - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = False # Use transaction

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": []
            }

            for reservation in reservations:
                warehouse_id = reservation['warehouse_id']
                product_id = reservation['product_id']
                quantity = reservation['quantity']

                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location,
                        total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                if inventory and inventory['available_quantity'] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))

                    result["reserved_items"].append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "warehouse_id": warehouse_id,
                        "warehouse_name": inventory['warehouse_name'],
                        "warehouse_location": inventory['warehouse_location']
                    })
                else:
                    result["failed_items"].append({
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "requested": quantity,
                        "available": inventory['available_quantity'] if inventory else 0,
                        "reason": "insufficient_stock" if inventory else "not_in_warehouse"
                    })

            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()
                result["success"] = False

            return result

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
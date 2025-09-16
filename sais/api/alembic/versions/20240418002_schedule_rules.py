"""Add scheduling rules and appointment metadata."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20240418002"
down_revision = "20240418001_initial"
branch_labels = None
depends_on = None


slot_status_enum = postgresql.ENUM(
    "FREE", "HOLD", "BOOKED", "BLOCKED", name="schedule_slot_status"
)
appointment_status_enum = postgresql.ENUM(
    "CONFIRMED", "RESCHEDULED", "CANCELLED", "NO_SHOW", "COMPLETED",
    name="appointment_status",
)
appointment_origin_enum = postgresql.ENUM(
    "WHATSAPP", "WEB", "STAFF", name="appointment_origin"
)


def upgrade() -> None:
    op.alter_column(
        "services",
        "duration_minutes",
        new_column_name="duration_min",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )
    op.add_column(
        "services",
        sa.Column("price_cents", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("services", "price_cents", server_default=None)

    slot_status_enum.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "schedule_slots",
        sa.Column(
            "status",
            slot_status_enum,
            nullable=False,
            server_default="FREE",
        ),
    )
    op.add_column(
        "schedule_slots",
        sa.Column("hold_expires_at", sa.DateTime(), nullable=True),
    )
    op.execute(
        "UPDATE schedule_slots SET status = 'BOOKED' WHERE is_booked = true"
    )
    op.drop_column("schedule_slots", "is_booked")
    op.alter_column("schedule_slots", "status", server_default=None)

    appointment_status_enum.create(op.get_bind(), checkfirst=True)
    appointment_origin_enum.create(op.get_bind(), checkfirst=True)
    op.execute(
        "UPDATE appointments SET status = 'CONFIRMED' WHERE status IS NULL OR status = 'scheduled'"
    )
    op.alter_column(
        "appointments",
        "status",
        type_=appointment_status_enum,
        existing_type=sa.String(length=32),
        postgresql_using="status::text::appointment_status",
        existing_nullable=False,
        server_default="CONFIRMED",
    )
    op.add_column(
        "appointments",
        sa.Column(
            "origin",
            appointment_origin_enum,
            nullable=False,
            server_default="WHATSAPP",
        ),
    )
    op.alter_column("appointments", "origin", server_default=None)


def downgrade() -> None:
    op.alter_column(
        "appointments",
        "status",
        type_=sa.String(length=32),
        existing_type=appointment_status_enum,
        server_default="scheduled",
        postgresql_using="status::text",
        existing_nullable=False,
    )
    op.drop_column("appointments", "origin")
    appointment_origin_enum.drop(op.get_bind(), checkfirst=True)
    appointment_status_enum.drop(op.get_bind(), checkfirst=True)

    op.add_column(
        "schedule_slots",
        sa.Column(
            "is_booked",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.execute(
        "UPDATE schedule_slots SET is_booked = true WHERE status IN ('BOOKED', 'BLOCKED')"
    )
    op.drop_column("schedule_slots", "hold_expires_at")
    op.drop_column("schedule_slots", "status")
    slot_status_enum.drop(op.get_bind(), checkfirst=True)

    op.drop_column("services", "price_cents")
    op.alter_column(
        "services",
        "duration_min",
        new_column_name="duration_minutes",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )

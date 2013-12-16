<?php get_header(); ?>


	<?php if (have_posts()) : ?>

		<?php while (have_posts()) : the_post(); ?>
		<div class="post">
			<div class="left">
				<h5><?php the_time('d'); ?></h5>
				<b><?php the_time('M y'); ?></b>
			</div>
			<div class="right">
				<h2><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
				<?php the_content('More...'); ?>
<br><?php comments_template(); ?>
			</div>
			<div class="clear"></div>
		</div>
		<?php endwhile; ?>



	<?php else : ?>

		<h2 class="center">Not Found</h2>

	<?php endif; ?>

<?php get_sidebar(); ?>
<?php get_footer(); ?>

